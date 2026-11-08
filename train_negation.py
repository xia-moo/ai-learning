import json
from pathlib import Path
from collections import Counter

project_dir = Path(__file__).resolve().parent
data_path = project_dir / "sentiment_negation_train.jsonl"

# 与原模型的类别顺序一致
label_to_id = {
    "负面": 0,
    "中性": 1,
    "正面": 2
}

new_texts = []
new_labels = []

with data_path.open("r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue

        item = json.loads(line)
        text = item["text"].strip()
        label = item["label"]

        if not text:
            raise ValueError(f"第 {line_number} 行文本不能为空")
        if label not in label_to_id:
            raise ValueError(f"第 {line_number} 行标签不合法")

        new_texts.append(text)
        new_labels.append(label_to_id[label])

print("新增训练样本数：", len(new_texts))
print("新增标签分布：", dict(Counter(new_labels)))

for text, label_id in zip(new_texts, new_labels):
    print(f"{label_id} | {text}")

positive_texts = [
    "商品质量很好",
    "这次购物很满意",
    "客服回答得很耐心",
    "操作简单又方便",
    "送货速度很快值得表扬",
    "包装精美让我很惊喜",
    "这个功能非常实用",
    "使用起来很舒服"
]
negative_texts = [
    "商品质量很差",
    "这次购物很失望",
    "客服态度让人生气",
    "操作复杂又麻烦",
    "等了很久还是没送到真烦",
    "包装破损让我很不满",
    "这个功能经常出错",
    "使用起来很不舒服"
]
neutral_texts = [
    "商品有三种颜色",
    "订单是在昨天提交的",
    "包装上印着产品型号",
    "说明书放在盒子里面",
    "这款产品重量是五百克",
    "客服工作时间是上午九点",
    "页面列出了产品尺寸",
    "我选择了蓝色的款式"
]

extra_data = [
    ("电池很耐用，我很满意", 2),
    ("收到实物后觉得物有所值", 2),
    ("运行流畅，没有出现卡顿", 2),
    ("这款商品的质量并不差", 2),

    ("电池很快就没电了，让我很失望", 0),
    ("收到实物后觉得不值这个价", 0),
    ("运行总是卡顿，实在难用", 0),
    ("这款商品的质量并不好", 0),

    ("电池容量标注在背面", 1),
    ("商品价格是两百元", 1),
    ("按下按钮可以切换模式", 1),
    ("盒子里有一根充电线", 1)
]

from sklearn.model_selection import train_test_split

all_texts = positive_texts + negative_texts + neutral_texts
all_labels = [2] * 8 + [0] * 8 + [1] * 8

# 与原实验使用完全相同的划分参数
train_texts, valid_texts, train_labels, valid_labels = train_test_split(
    all_texts,
    all_labels,
    test_size=0.25,
    random_state=42,
    stratify=all_labels
)

# 恢复原来的 30 条训练数据
original_train_texts = train_texts + [
    text for text, label in extra_data
]
original_train_labels = train_labels + [
    label for text, label in extra_data
]

# 加入 8 条否定表达样例
combined_texts = original_train_texts + new_texts
combined_labels = original_train_labels + new_labels

# 检查完全重复及训练、验证文本重叠
normalized_train = [text.strip() for text in combined_texts]
normalized_valid = {text.strip() for text in valid_texts}

assert len(set(normalized_train)) == len(normalized_train), "训练文本重复"
assert not set(normalized_train) & normalized_valid, "训练与验证文本重叠"

print("\n原训练样本数：", len(original_train_texts))
print("合并后训练样本数：", len(combined_texts))
print("合并后标签分布：", dict(Counter(combined_labels)))
print("验证样本数：", len(valid_texts))
import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModel

checkpoint_path = project_dir / "bert_sentiment_best.pt"
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

class_names = checkpoint["class_names"]

# 确保新增数据的标签编号与模型一致
assert {
    name: index for index, name in enumerate(class_names)
} == label_to_id, "标签编号与检查点不一致"

assert checkpoint["pooling"] == "masked_mean", "池化方式不一致"

tokenizer = AutoTokenizer.from_pretrained(
    checkpoint["model_name"],
    local_files_only=True
)

config = AutoConfig.from_pretrained(
    checkpoint["model_name"],
    local_files_only=True
)

bert = AutoModel.from_config(config)
bert.load_state_dict(checkpoint["bert_state"])

classifier = nn.Linear(
    checkpoint["hidden_size"],
    len(class_names)
)
classifier.load_state_dict(checkpoint["classifier_state"])

# 本次准备同时训练 BERT 和分类层
for parameter in bert.parameters():
    parameter.requires_grad = True

for parameter in classifier.parameters():
    parameter.requires_grad = True

bert.eval()
classifier.eval()

print("\n已加载原检查点：", checkpoint_path.name)
print("类别顺序：", class_names)
print("BERT 参数类型：", next(bert.parameters()).dtype)
print("BERT 参数是否全部可训练：",
      all(p.requires_grad for p in bert.parameters()))
# 沿用原训练实验的长度上限，保持前后比较条件一致
TRAIN_MAX_LENGTH = checkpoint["max_length"]
criterion = nn.CrossEntropyLoss()


def forward_logits(batch):
    token_vectors = bert(**batch).last_hidden_state

    mask = batch["attention_mask"].unsqueeze(-1).to(
        token_vectors.dtype
    )

    sentence_vectors = (
        (token_vectors * mask).sum(dim=1)
        / mask.sum(dim=1).clamp(min=1)
    )

    return classifier(sentence_vectors)


valid_batch = tokenizer(
    valid_texts,
    padding=True,
    truncation=True,
    max_length=TRAIN_MAX_LENGTH,
    return_tensors="pt"
)

valid_y = torch.tensor(valid_labels, dtype=torch.long)


def evaluate_validation():
    bert.eval()
    classifier.eval()

    with torch.no_grad():
        logits = forward_logits(valid_batch)
        loss = criterion(logits, valid_y).item()
        predictions = logits.argmax(dim=1)

        accuracy = (
            (predictions == valid_y).float().mean().item()
        )

    return loss, accuracy


before_loss, before_accuracy = evaluate_validation()

print("本次训练与验证的长度上限：", TRAIN_MAX_LENGTH)
print(f"继续训练前验证损失：{before_loss:.4f}")
print(f"继续训练前验证准确率：{before_accuracy:.1%}")
from copy import deepcopy
from torch.utils.data import DataLoader

torch.manual_seed(42)

encoded_train = tokenizer(
    combined_texts,
    padding=True,
    truncation=True,
    max_length=TRAIN_MAX_LENGTH,
    return_tensors="pt"
)

# 保留 tokenizer 返回的所有输入字段
train_dataset = [
    {
        **{
            key: value[index]
            for key, value in encoded_train.items()
        },
        "labels": torch.tensor(label, dtype=torch.long)
    }
    for index, label in enumerate(combined_labels)
]

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

# 新优化器；学习率是本次实验的起始设置
optimizer = torch.optim.AdamW(
    [
        {"params": bert.parameters(), "lr": 1e-5},
        {"params": classifier.parameters(), "lr": 1e-4}
    ],
    weight_decay=0.01
)

best_epoch = 0
best_valid_loss = before_loss
best_state = {
    "bert": deepcopy(bert.state_dict()),
    "classifier": deepcopy(classifier.state_dict())
}

print("\n每轮批次数：", len(train_loader))

for epoch in range(1, 4):
    bert.train()
    classifier.train()

    total_loss = 0.0
    total_samples = 0

    for batch in train_loader:
        labels = batch.pop("labels")

        optimizer.zero_grad()
        logits = forward_logits(batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    valid_loss, valid_accuracy = evaluate_validation()

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_epoch = epoch
        best_state = {
            "bert": deepcopy(bert.state_dict()),
            "classifier": deepcopy(classifier.state_dict())
        }

    print(
        f"第 {epoch} 轮 | "
        f"训练平均损失={total_loss / total_samples:.4f} | "
        f"验证损失={valid_loss:.4f} | "
        f"验证准确率={valid_accuracy:.1%}"
    )

bert.load_state_dict(best_state["bert"])
classifier.load_state_dict(best_state["classifier"])

restored_loss, restored_accuracy = evaluate_validation()

print("\n选中的轮数：", best_epoch)
print(f"恢复后验证损失：{restored_loss:.4f}")
print(f"恢复后验证准确率：{restored_accuracy:.1%}")
output_path = project_dir / "bert_sentiment_negation_v1.pt"

# 保留原检查点中的模型结构、类别顺序和池化等配置
new_checkpoint = dict(checkpoint)

new_checkpoint.update({
    "bert_state": bert.state_dict(),
    "classifier_state": classifier.state_dict(),
    "max_length": TRAIN_MAX_LENGTH,
    "best_epoch": best_epoch,
    "best_valid_loss": restored_loss,
    "experiment": "negation_finetune_v1",
    "parent_checkpoint": checkpoint_path.name,
    "train_sample_count": len(combined_texts),
    "valid_sample_count": len(valid_texts),
    "before_valid_loss": before_loss
})

# 仅创建新文件；如果同名文件已存在，会报错，避免覆盖
with output_path.open("xb") as file:
    torch.save(new_checkpoint, file)

print("新模型已保存：", output_path)