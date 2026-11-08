from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from time import perf_counter
from statistics import mean, median
from torch.ao.quantization import quantize_dynamic
import hashlib

# 当前服务允许的输入长度上限
checkpoint_path = (
    Path(__file__).resolve().parent / "bert_sentiment_negation_v1.pt"
)

USE_INT8 = False
INFERENCE_MAX_LENGTH = 64
checkpoint_hasher = hashlib.sha256()

with checkpoint_path.open("rb") as file:
    while True:
        chunk = file.read(1024 * 1024)
        if not chunk:
            break
        checkpoint_hasher.update(chunk)

CHECKPOINT_SHA256 = checkpoint_hasher.hexdigest()
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

model_name = checkpoint["model_name"]
class_names = checkpoint["class_names"]

# 使用之前下载到本机的分词器和配置
tokenizer = AutoTokenizer.from_pretrained(
    model_name, local_files_only=True
)
config = AutoConfig.from_pretrained(
    model_name, local_files_only=True
)

# 先创建模型结构，再加载自己训练好的参数
bert = AutoModel.from_config(config)
bert.load_state_dict(checkpoint["bert_state"])

bert.eval()

if USE_INT8:
    bert = quantize_dynamic(
        bert,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=False
    )

print("当前推理模式：", "INT8" if USE_INT8 else "FP32")

classifier = nn.Linear(
    checkpoint["hidden_size"],
    len(class_names)
)
classifier.load_state_dict(checkpoint["classifier_state"])

if checkpoint["pooling"] != "masked_mean":
    raise ValueError("当前脚本只支持 masked_mean 池化")

bert.eval()
classifier.eval()

# 用未参与前面训练和验证的文本做小规模检查
texts = [
    "这个产品好用",
    "这个产品不好用",

    "我对这次购物满意",
    "我对这次购物不满意",

    "这个功能稳定",
    "这个功能不稳定"
]

# 每一对都是：正面、负面
test_labels = [2, 0, 2, 0, 2, 0]

batch = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=checkpoint["max_length"],
    return_tensors="pt"
)

with torch.no_grad():
    vectors = bert(**batch).last_hidden_state

    mask = batch["attention_mask"].unsqueeze(-1)
    mask = mask.to(vectors.dtype)

    features = (
        (vectors * mask).sum(dim=1)
        / mask.sum(dim=1).clamp(min=1)
    )

    logits = classifier(features)
    probabilities = torch.softmax(logits, dim=-1)
    predictions = logits.argmax(dim=-1)

print("已加载检查点：", checkpoint_path)
print("保存的最佳轮数：", checkpoint["best_epoch"])

for text, prediction, probs in zip(
    texts,
    predictions.tolist(),
    probabilities.tolist()
):
    print(f"\n{text} | 预测：{class_names[prediction]}")
    print(" | ".join(
        f"{name}：{prob:.3f}"
        for name, prob in zip(class_names, probs)
    ))
correct = sum(
    prediction == target
    for prediction, target in zip(
        predictions.tolist(), test_labels
    )
)

print(f"\n新文本正确数：{correct}/{len(test_labels)}")
def predict_one(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("请输入非空文本")

    text = text.strip()
    limit = INFERENCE_MAX_LENGTH

    # 不截断，统计完整长度；包含特殊 token
    original_ids = tokenizer(
        text,
        truncation=False
    )["input_ids"]

    original_count = len(original_ids)

    # 按当前模型配置生成实际输入
    batch = tokenizer(
        text,
        truncation=True,
        max_length=limit,
        return_tensors="pt"
    )

    # 使用 mask 统计有效位置，排除可能的填充
    input_count = int(batch["attention_mask"].sum().item())

    bert.eval()
    classifier.eval()

    with torch.no_grad():
        vectors = bert(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        mask = mask.to(vectors.dtype)

        features = (
            (vectors * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

        logits = classifier(features)
        probs = torch.softmax(logits, dim=-1)[0]
        predicted_id = probs.argmax().item()

    return {
        "text": text,
        "label": class_names[predicted_id],
        "probabilities": dict(zip(class_names, probs.tolist())),
        "max_length": limit,
        "original_token_count": original_count,
        "input_token_count": input_count,
        "truncated": original_count > input_count
    }

def predict_batch(texts):
    if not isinstance(texts, list) or not texts:
        raise ValueError("请输入非空文本列表")

    if any(
        not isinstance(text, str) or not text.strip()
        for text in texts
    ):
        raise ValueError("每条文本都必须是非空字符串")

    cleaned_texts = [text.strip() for text in texts]
    
    original_encoded = tokenizer(
    cleaned_texts,
    padding=False,
    truncation=False
    )
    original_counts = [
        len(ids) for ids in original_encoded["input_ids"]
    ]

    batch = tokenizer(
        cleaned_texts,
        padding=True,
        truncation=True,
        max_length=INFERENCE_MAX_LENGTH,
        return_tensors="pt"
    )
    input_counts = batch["attention_mask"].sum(dim=1).tolist()
    bert.eval()
    classifier.eval()

    with torch.no_grad():
        vectors = bert(**batch).last_hidden_state

        mask = batch["attention_mask"].unsqueeze(-1)
        mask = mask.to(vectors.dtype)

        features = (
            (vectors * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

        logits = classifier(features)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

    results = []

    for i, text in enumerate(cleaned_texts):
        predicted_id = predictions[i].item()

        results.append({
            "text": text,
            "label": class_names[predicted_id],
            "probabilities": dict(
                zip(class_names, probabilities[i].tolist())
            ),
            "max_length": INFERENCE_MAX_LENGTH,
            "original_token_count": original_counts[i],
            "input_token_count": input_counts[i],
            "truncated": original_counts[i] > input_counts[i]
        })

    return results

def predict_grouped(texts, batch_size=4):
    if not isinstance(texts, list) or not texts:
        raise ValueError("请输入非空文本列表")

    if any(
        not isinstance(text, str) or not text.strip()
        for text in texts
    ):
        raise ValueError("每条文本都必须是非空字符串")

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")

    cleaned_texts = [text.strip() for text in texts]

    # 使用与预测相同的截断规则，计算实际 token 长度
    encoded = tokenizer(
        cleaned_texts,
        padding=False,
        truncation=True,
        max_length=INFERENCE_MAX_LENGTH
    )

    lengths = [len(ids) for ids in encoded["input_ids"]]

    # 排序的是原始索引，便于之后恢复顺序
    sorted_indices = sorted(
        range(len(cleaned_texts)),
        key=lambda index: lengths[index]
    )

    results = [None] * len(cleaned_texts)

    for start in range(0, len(sorted_indices), batch_size):
        indices = sorted_indices[start:start + batch_size]
        batch_texts = [cleaned_texts[index] for index in indices]

        batch_results = predict_batch(batch_texts)

        # 将结果放回原始位置
        for index, result in zip(indices, batch_results):
            results[index] = result

    return results

def predict_in_order(texts, batch_size=2):
    results = []

    for start in range(0, len(texts), batch_size):
        results.extend(
            predict_batch(texts[start:start + batch_size])
        )

    return results

def predict_grouped_once(texts, batch_size=4):
    if not isinstance(texts, list) or not texts:
        raise ValueError("请输入非空文本列表")

    if any(
        not isinstance(text, str) or not text.strip()
        for text in texts
    ):
        raise ValueError("每条文本都必须是非空字符串")

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")

    cleaned_texts = [text.strip() for text in texts]

    # 只分词一次，暂时保留不同长度的列表
    encoded = tokenizer(
        cleaned_texts,
        padding=False,
        truncation=True,
        max_length=INFERENCE_MAX_LENGTH
    )

    sorted_indices = sorted(
        range(len(cleaned_texts)),
        key=lambda index: len(encoded["input_ids"][index])
    )

    results = [None] * len(cleaned_texts)

    bert.eval()
    classifier.eval()

    with torch.no_grad():
        for start in range(0, len(sorted_indices), batch_size):
            indices = sorted_indices[start:start + batch_size]

            # 取出这一批已经分词的结果，保留所有输入字段
            items = [
                {
                    key: values[index]
                    for key, values in encoded.items()
                }
                for index in indices
            ]

            # 只补齐当前小批次，并转换成张量
            batch = tokenizer.pad(
                items,
                padding=True,
                return_tensors="pt"
            )

            vectors = bert(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            mask = mask.to(vectors.dtype)

            features = (
                (vectors * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1)
            )

            logits = classifier(features)
            probabilities = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)

            for index, prediction, probs in zip(
                indices,
                predictions.tolist(),
                probabilities.tolist()
            ):
                results[index] = {
                    "text": cleaned_texts[index],
                    "label": class_names[prediction],
                    "probabilities": dict(zip(class_names, probs))
                }

    return results

def profile_once(text):
    # 当前模型在 CPU 上运行，可以直接用 perf_counter 计时
    start = perf_counter()

    batch = tokenizer(
        text,
        truncation=True,
        max_length=checkpoint["max_length"],
        return_tensors="pt"
    )
    after_tokenize = perf_counter()

    with torch.no_grad():
        vectors = bert(**batch).last_hidden_state
        after_bert = perf_counter()

        mask = batch["attention_mask"].unsqueeze(-1)
        mask = mask.to(vectors.dtype)

        features = (
            (vectors * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

        logits = classifier(features)
        probabilities = torch.softmax(logits, dim=-1).tolist()
        finished = perf_counter()

    return {
        "分词": (after_tokenize - start) * 1000,
        "BERT": (after_bert - after_tokenize) * 1000,
        "池化与分类": (finished - after_bert) * 1000
    }

def measure_text_limit(text, limit):
    start = perf_counter()

    batch = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=limit,
        return_tensors="pt"
    )

    with torch.no_grad():
        vectors = bert(**batch).last_hidden_state

        mask = batch["attention_mask"].unsqueeze(-1)
        mask = mask.to(vectors.dtype)

        features = (
            (vectors * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

        logits = classifier(features)
        prediction = logits.argmax(dim=-1).item()

    elapsed_ms = (perf_counter() - start) * 1000
    return elapsed_ms, batch["input_ids"].shape[1], prediction


if __name__ == "__main__":
    batch_results = predict_batch([
    "好用",
    "这个功能不好用",
    "我对这次购物很满意",
    "我买的是白色版本"
    ])
    
    for result in batch_results:
        print(f"{result['text']} → {result['label']}")
    
    examples = [
    "这个产品操作起来很方便，我对这次购物非常满意",
    "差",
    "商品有三种颜色",
    "这个功能经常出错，使用过程中让我很失望",
    "好用",
    "我买的是白色版本"
    ]

    grouped_results = predict_grouped(examples, batch_size=2)

    # 检查返回文本是否仍与输入顺序一致
    assert [item["text"] for item in grouped_results] == examples

    for item in grouped_results:
        print(f"{item['text']} → {item['label']}")

    print("返回顺序检查通过")
    
    encoded = tokenizer(
        examples,
        padding=False,
        truncation=True,
        max_length=checkpoint["max_length"]
    )

    lengths = [len(ids) for ids in encoded["input_ids"]]
    sorted_lengths = sorted(lengths)
    batch_size = 2
    valid_tokens = sum(lengths)

    print("原始 token 长度：", lengths)
    print("排序后的长度：", sorted_lengths)

    for name, current_lengths in [
        ("原顺序分批", lengths),
        ("按长度分批", sorted_lengths)
    ]:
        total_positions = 0

        for start in range(0, len(current_lengths), batch_size):
            group = current_lengths[start:start + batch_size]

            # 每批补齐到该批最长的长度
            total_positions += len(group) * max(group)

        padding_count = total_positions - valid_tokens

        print(
            f"{name} | "
            f"有效 token={valid_tokens} | "
            f"填充位置={padding_count} | "
            f"总位置={total_positions} | "
            f"填充比例={padding_count / total_positions:.1%}"
        )
    
    methods = [
        ("原顺序分批", predict_in_order),
        ("分组但重复分词", predict_grouped),
        ("分组且只分词一次", predict_grouped_once)
    ]

    # 每种方法分别预热
    for _, function in methods:
        for _ in range(3):
            function(examples, batch_size=2)

    timings = {name: [] for name, _ in methods}

    for repeat in range(30):
        # 轮换顺序，让每种方法都有机会先运行
        offset = repeat % len(methods)
        run_order = methods[offset:] + methods[:offset]

        for name, function in run_order:
            start = perf_counter()
            results = function(examples, batch_size=2)
            elapsed_ms = (perf_counter() - start) * 1000

            timings[name].append(elapsed_ms)

    for name, values in timings.items():
        print(
            f"{name} | "
            f"平均处理六条耗时={mean(values):.1f} 毫秒 | "
            f"P50={median(values):.1f} 毫秒"
        )
    
    old_results = predict_grouped(examples, batch_size=2)
    new_results = predict_grouped_once(examples, batch_size=2)

    assert [item["text"] for item in new_results] == examples

    for old, new in zip(old_results, new_results):
        old_probs = torch.tensor(list(old["probabilities"].values()))
        new_probs = torch.tensor(list(new["probabilities"].values()))

        assert old["label"] == new["label"]
        assert torch.allclose(
            old_probs, new_probs, atol=1e-6, rtol=1e-5
        )

    print("顺序、类别和概率一致性检查通过")
    
    bert.eval()
    classifier.eval()

    profile_text = "这个产品很好用"

    # 预热
    for _ in range(3):
        profile_once(profile_text)

    records = [profile_once(profile_text) for _ in range(30)]

    averages = {
        name: mean(record[name] for record in records)
        for name in records[0]
    }
    total = sum(averages.values())

    for name, elapsed in averages.items():
        print(
            f"{name}：平均 {elapsed:.3f} ms"
            f" | 占比 {elapsed / total:.1%}"
        )

    print(f"三个阶段合计：{total:.3f} ms")
    
    long_text = (
        "这款产品有多种颜色，包装里附带说明书和充电线，"
        "外观与页面描述一致，但是用了两天就坏了，我非常失望。"
    )

    full_ids = tokenizer(
        long_text,
        truncation=False
    )["input_ids"]

    limited_ids = tokenizer(
        long_text,
        truncation=True,
        max_length=checkpoint["max_length"]
    )["input_ids"]

    print("完整 token 数：", len(full_ids))
    print("实际输入 token 数：", len(limited_ids))
    print(
        "截断后保留的文本：",
        tokenizer.decode(limited_ids, skip_special_tokens=True)
    )
    
    bert.eval()
    classifier.eval()

    for limit in [32, 64]:
        batch = tokenizer(
            long_text,
            padding=True,
            truncation=True,
            max_length=limit,
            return_tensors="pt"
        )

        with torch.no_grad():
            vectors = bert(**batch).last_hidden_state

            mask = batch["attention_mask"].unsqueeze(-1)
            mask = mask.to(vectors.dtype)

            features = (
                (vectors * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1)
            )

            logits = classifier(features)
            probs = torch.softmax(logits, dim=-1)[0]
            prediction = probs.argmax().item()

        print(
            f"\n长度上限：{limit}"
            f" | 实际长度：{batch['input_ids'].shape[1]}"
            f" | 预测：{class_names[prediction]}"
        )

        for name, probability in zip(class_names, probs.tolist()):
            print(f"{name}：{probability:.3f}")
    
    bert.eval()
    classifier.eval()

    for limit in [32, 64]:
        for _ in range(3):
            measure_text_limit(long_text, limit)

    timings = {32: [], 64: []}
    details = {}

    for repeat in range(30):
        limits = [32, 64] if repeat % 2 == 0 else [64, 32]

        for limit in limits:
            elapsed, length, prediction = measure_text_limit(
                long_text, limit
            )
            timings[limit].append(elapsed)
            details[limit] = (length, prediction)

    for limit in [32, 64]:
        length, prediction = details[limit]

        print(
            f"上限={limit} | 实际长度={length} | "
            f"预测={class_names[prediction]} | "
            f"平均耗时={mean(timings[limit]):.1f} ms | "
            f"P50={median(timings[limit]):.1f} ms"
        )
    
    while True:
        text = input("\n请输入评论，输入 exit 退出：").strip()

        if text.lower() == "exit":
            print("已退出")
            break

        if not text:
            print("输入不能为空，请重新输入。")
            continue

        result = predict_one(text)

        print("预测类别：", result["label"])

        for name, probability in result["probabilities"].items():
            print(f"{name}：{probability:.3f}")