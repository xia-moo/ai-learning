from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.ao.quantization import quantize_dynamic
from torch.ao.nn.quantized.dynamic import Linear as QuantizedLinear

checkpoint_path = (
    Path(__file__).resolve().parent / "bert_sentiment_best.pt"
)

saved = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

class_names = saved["class_names"]
max_length = saved["max_length"]

if saved["pooling"] != "masked_mean":
    raise ValueError("当前脚本只支持 masked_mean 池化")

tokenizer = AutoTokenizer.from_pretrained(
    saved["model_name"], local_files_only=True
)
config = AutoConfig.from_pretrained(
    saved["model_name"], local_files_only=True
)

bert_fp32 = AutoModel.from_config(config)
bert_fp32.load_state_dict(saved["bert_state"])
bert_fp32.eval()

classifier = nn.Linear(
    saved["hidden_size"], len(class_names)
)
classifier.load_state_dict(saved["classifier_state"])
classifier.eval()

# 参数已加载，释放检查点字典占用的内存
del saved

# 创建量化副本，原始 BERT 保持不变
bert_int8 = quantize_dynamic(
    bert_fp32,
    {nn.Linear},
    dtype=torch.qint8,
    inplace=False
)
bert_int8.eval()

converted_count = sum(
    isinstance(layer, QuantizedLinear)
    for layer in bert_int8.modules()
)
print("已量化的线性层数量：", converted_count)


def predict_with(encoder, batch):
    with torch.no_grad():
        vectors = encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        mask = mask.to(vectors.dtype)

        features = (
            (vectors * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

        logits = classifier(features)
        return torch.softmax(logits, dim=-1)


# 复用原来的六条验证文本，比较量化前后表现
texts = [
    "使用起来很舒服",
    "页面列出了产品尺寸",
    "这个功能经常出错",
    "订单是在昨天提交的",
    "操作复杂又麻烦",
    "客服回答得很耐心"
]
labels = torch.tensor([2, 1, 0, 1, 0, 2])

batch = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="pt"
)

probs_fp32 = predict_with(bert_fp32, batch)
probs_int8 = predict_with(bert_int8, batch)

preds_fp32 = probs_fp32.argmax(dim=1)
preds_int8 = probs_int8.argmax(dim=1)

for text, target, before, after in zip(
    texts, labels.tolist(),
    preds_fp32.tolist(), preds_int8.tolist()
):
    print(
        f"{text} | 真实：{class_names[target]}"
        f" | FP32：{class_names[before]}"
        f" | INT8：{class_names[after]}"
    )

print("FP32 正确数：", (preds_fp32 == labels).sum().item(), "/6")
print("INT8 正确数：", (preds_int8 == labels).sum().item(), "/6")
print("类别改变数：", (preds_fp32 != preds_int8).sum().item())
print(
    "最大概率绝对差异：",
    (probs_fp32 - probs_int8).abs().max().item()
)
import io
from time import perf_counter
from statistics import mean, median


def serialized_size_mib(encoder):
    # 同时包含 BERT 和分类层，与实际预测模型对应
    with io.BytesIO() as buffer:
        torch.save(
            {
                "bert": encoder.state_dict(),
                "classifier": classifier.state_dict()
            },
            buffer
        )
        return buffer.tell() / (1024 ** 2)


size_fp32 = serialized_size_mib(bert_fp32)
size_int8 = serialized_size_mib(bert_int8)

print(f"\nFP32 参数序列化大小：{size_fp32:.1f} MiB")
print(f"INT8 参数序列化大小：{size_int8:.1f} MiB")
print(f"大小减少：{1 - size_int8 / size_fp32:.1%}")

models = [
    ("FP32", bert_fp32),
    ("INT8", bert_int8)
]

# 两种模型分别预热
for _, encoder in models:
    for _ in range(3):
        predict_with(encoder, batch)

timings = {name: [] for name, _ in models}

for repeat in range(30):
    run_order = models if repeat % 2 == 0 else models[::-1]

    for name, encoder in run_order:
        start = perf_counter()
        predict_with(encoder, batch)
        elapsed_ms = (perf_counter() - start) * 1000

        timings[name].append(elapsed_ms)

for name, values in timings.items():
    print(
        f"{name} | "
        f"平均耗时={mean(values):.1f} ms | "
        f"P50={median(values):.1f} ms"
    )

speedup = mean(timings["FP32"]) / mean(timings["INT8"])
print(f"平均耗时加速比：{speedup:.2f} 倍")