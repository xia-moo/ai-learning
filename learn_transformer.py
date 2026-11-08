import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel

model_name = "google-bert/bert-base-chinese"

tokenizer = AutoTokenizer.from_pretrained(model_name)
bert = AutoModel.from_pretrained(model_name)

# 冻结 BERT 参数
for parameter in bert.parameters():
    parameter.requires_grad = False

# 关闭训练模式下的 Dropout
bert.eval()

texts = [
    "这个产品很好用",
    "体验太差了"
]

batch = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=32,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = bert(**batch)
    token_vectors = outputs.last_hidden_state

    # 排除填充位置，计算句子向量
    mask = batch["attention_mask"].unsqueeze(-1)
    mask = mask.to(token_vectors.dtype)

    sentence_vectors = (
        (token_vectors * mask).sum(dim=1)
        / mask.sum(dim=1).clamp(min=1)
    )

print("输入 ID 形状：", batch["input_ids"].shape)
print("token 向量形状：", token_vectors.shape)
print("句子向量形状：", sentence_vectors.shape)
print("句子向量是否需要梯度：", sentence_vectors.requires_grad)
def extract_features(texts):
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )

    bert.eval()

    with torch.no_grad():
        vectors = bert(**batch).last_hidden_state

        mask = batch["attention_mask"].unsqueeze(-1)
        mask = mask.to(vectors.dtype)

        features = (
            (vectors * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

    return features


# 三类各 8 条教学数据；与前面练习使用相同的划分
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

all_texts = positive_texts + negative_texts + neutral_texts
all_labels = [2] * 8 + [0] * 8 + [1] * 8

train_texts, valid_texts, train_labels, valid_labels = train_test_split(
    all_texts,
    all_labels,
    test_size=0.25,
    random_state=42,
    stratify=all_labels
)


train_features = extract_features(train_texts)
valid_features = extract_features(valid_texts)

train_y = torch.tensor(train_labels, dtype=torch.long)
valid_y = torch.tensor(valid_labels, dtype=torch.long)

print("训练特征：", train_features.shape)
print("验证特征：", valid_features.shape)
print("训练标签：", train_y.shape)
print("验证标签：", valid_y.shape)
from torch import nn
from copy import deepcopy
# 新增训练样本：(文本, 标签)
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

# 创建新列表，避免重复运行时不断追加到原训练集
augmented_texts = train_texts + [text for text, label in extra_data]
augmented_labels = train_labels + [label for text, label in extra_data]

# BERT 仍然冻结，重新提取扩充后的训练特征
train_features = extract_features(augmented_texts)
train_y = torch.tensor(augmented_labels, dtype=torch.long)

print("扩充后训练特征：", train_features.shape)
print("扩充后训练标签：", train_y.shape)

torch.manual_seed(42)

classifier = nn.Linear(768, 3)

# 优化器只管理分类层，不包含 BERT
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

best_loss = float("inf")
best_state = None
best_epoch = 0
wait = 0
patience = 20

for epoch in range(1, 101):
    classifier.train()

    logits = classifier(train_features)
    loss = loss_fn(logits, train_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 用更新后的参数评估训练集和验证集
    classifier.eval()
    with torch.no_grad():
        train_logits = classifier(train_features)
        valid_logits = classifier(valid_features)

        train_loss = loss_fn(train_logits, train_y).item()
        valid_loss = loss_fn(valid_logits, valid_y).item()

        train_acc = (
            train_logits.argmax(dim=1) == train_y
        ).float().mean().item()

        valid_acc = (
            valid_logits.argmax(dim=1) == valid_y
        ).float().mean().item()

    if valid_loss < best_loss:
        best_loss = valid_loss
        best_epoch = epoch
        best_state = deepcopy(classifier.state_dict())
        wait = 0
    else:
        wait += 1

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"第 {epoch:3d} 轮 | "
            f"训练损失={train_loss:.4f} | "
            f"验证损失={valid_loss:.4f} | "
            f"训练准确率={train_acc:.3f} | "
            f"验证准确率={valid_acc:.3f}"
        )

    if wait >= patience:
        print(f"第 {epoch} 轮触发早停")
        break

# 恢复验证损失最低时的分类层参数
classifier.load_state_dict(best_state)
classifier.eval()

with torch.no_grad():
    predictions = classifier(valid_features).argmax(dim=1)
    best_acc = (predictions == valid_y).float().mean().item()

print("最佳轮数：", best_epoch)
print(f"最佳验证损失：{best_loss:.4f}")
print(f"恢复后验证准确率：{best_acc:.3f}")
class_names = ["负面", "中性", "正面"]

classifier.eval()

with torch.no_grad():
    logits = classifier(valid_features)
    predictions = logits.argmax(dim=1)

for text, target, prediction in zip(
    valid_texts,
    valid_y.tolist(),
    predictions.tolist()
):
    result = "正确" if target == prediction else "错误"

    print(
        f"{result} | {text}"
        f" | 真实：{class_names[target]}"
        f" | 预测：{class_names[prediction]}"
    )
    
from sklearn.metrics import classification_report

classifier.eval()

with torch.no_grad():
    predictions = classifier(valid_features).argmax(dim=1)

print(classification_report(
    valid_y.tolist(),
    predictions.tolist(),
    labels=[0, 1, 2],
    target_names=["负面", "中性", "正面"],
    digits=3,
    zero_division=0
))
classifier.eval()

with torch.no_grad():
    probabilities = torch.softmax(
        classifier(valid_features), dim=1
    )

for text, target, probs in zip(
    valid_texts, valid_y.tolist(), probabilities.tolist()
):
    if target == 0:
        print(text)
        print(
            f"负面：{probs[0]:.3f} | "
            f"中性：{probs[1]:.3f} | "
            f"正面：{probs[2]:.3f}"
        )
        
# 解除 BERT 参数冻结
for parameter in bert.parameters():
    parameter.requires_grad = True

bert.train()
classifier.train()

# 清掉之前可能残留的梯度
bert.zero_grad(set_to_none=True)
classifier.zero_grad(set_to_none=True)

# 先取两条训练文本做检查
probe_batch = tokenizer(
    train_texts[:2],
    padding=True,
    truncation=True,
    max_length=32,
    return_tensors="pt"
)
probe_y = torch.tensor(train_labels[:2], dtype=torch.long)

# 这里不能放在 no_grad 中
vectors = bert(**probe_batch).last_hidden_state

mask = probe_batch["attention_mask"].unsqueeze(-1)
mask = mask.to(vectors.dtype)

features = (
    (vectors * mask).sum(dim=1)
    / mask.sum(dim=1).clamp(min=1)
)

logits = classifier(features)
loss = nn.CrossEntropyLoss()(logits, probe_y)
loss.backward()

print("句子向量是否需要梯度：", features.requires_grad)
print(
    "BERT 是否收到梯度：",
    any(p.grad is not None for p in bert.parameters())
)
print("分类层是否收到梯度：", classifier.weight.grad is not None)
optimizer = torch.optim.AdamW(
    [
        {"params": bert.parameters(), "lr": 2e-5},
        {"params": classifier.parameters(), "lr": 1e-3}
    ],
    weight_decay=0.01
)

bert.train()
classifier.train()

# 清除刚才检查时产生的梯度
optimizer.zero_grad(set_to_none=True)

# 重新进行前向计算
vectors = bert(**probe_batch).last_hidden_state

mask = probe_batch["attention_mask"].unsqueeze(-1)
mask = mask.to(vectors.dtype)

features = (
    (vectors * mask).sum(dim=1)
    / mask.sum(dim=1).clamp(min=1)
)

logits = classifier(features)
loss = nn.CrossEntropyLoss()(logits, probe_y)

# 保存分类层更新前的权重，方便检查
weight_before = classifier.weight.detach().clone()

loss.backward()
optimizer.step()

print("本次更新前损失：", loss.item())
print("BERT 学习率：", optimizer.param_groups[0]["lr"])
print("分类层学习率：", optimizer.param_groups[1]["lr"])
print(
    "分类层权重是否发生变化：",
    not torch.equal(weight_before, classifier.weight.detach())
)
from torch.utils.data import TensorDataset, DataLoader

# 使用扩充后的 30 条训练文本
finetune_batch = tokenizer(
    augmented_texts,
    padding=True,
    truncation=True,
    max_length=32,
    return_tensors="pt"
)

finetune_y = torch.tensor(
    augmented_labels,
    dtype=torch.long
)

# 每个样本包含：token ID、mask、真实标签
train_dataset = TensorDataset(
    finetune_batch["input_ids"],
    finetune_batch["attention_mask"],
    finetune_y
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

print("训练样本数：", len(train_dataset))
print("每轮批次数：", len(train_loader))

for batch_number, (ids, mask, labels) in enumerate(
    train_loader, start=1
):
    print(
        f"第 {batch_number} 批 | "
        f"ID：{tuple(ids.shape)} | "
        f"mask：{tuple(mask.shape)} | "
        f"标签：{tuple(labels.shape)}"
    )
# 每次调用都重新运行当前 BERT，不使用缓存特征
def classify_batch(ids, attention_mask):
    vectors = bert(
        input_ids=ids,
        attention_mask=attention_mask
    ).last_hidden_state

    mask = attention_mask.unsqueeze(-1).to(vectors.dtype)

    features = (
        (vectors * mask).sum(dim=1)
        / mask.sum(dim=1).clamp(min=1)
    )

    return classifier(features)


# 验证文本只需分词一次，但向量每轮必须重新计算
finetune_valid_batch = tokenizer(
    valid_texts,
    padding=True,
    truncation=True,
    max_length=32,
    return_tensors="pt"
)
finetune_valid_y = torch.tensor(
    valid_labels, dtype=torch.long
)

fine_tune_steps = 0
from copy import deepcopy

best_valid_loss = float("inf")
best_epoch = 0
best_checkpoint = None

for epoch in range(1, 4):
    bert.train()
    classifier.train()

    total_loss = 0.0
    sample_count = 0

    for ids, attention_mask, labels in train_loader:
        optimizer.zero_grad(set_to_none=True)

        logits = classify_batch(ids, attention_mask)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()
        fine_tune_steps += 1

        # 按样本数累计，正确处理最后只有 2 条的批次
        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        sample_count += batch_size

    bert.eval()
    classifier.eval()

    with torch.no_grad():
        valid_logits = classify_batch(
            finetune_valid_batch["input_ids"],
            finetune_valid_batch["attention_mask"]
        )

        valid_loss = loss_fn(
            valid_logits, finetune_valid_y
        ).item()
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch

            best_checkpoint = {
                "bert": deepcopy(bert.state_dict()),
                "classifier": deepcopy(classifier.state_dict())
            }
            
        valid_acc = (
            valid_logits.argmax(dim=1) == finetune_valid_y
        ).float().mean().item()

    print(
        f"第 {epoch} 轮 | "
        f"本段累计更新={fine_tune_steps} | "
        f"训练平均损失={total_loss / sample_count:.4f} | "
        f"验证损失={valid_loss:.4f} | "
        f"验证准确率={valid_acc:.3f}"
    )
bert.load_state_dict(best_checkpoint["bert"])
classifier.load_state_dict(best_checkpoint["classifier"])

bert.eval()
classifier.eval()

with torch.no_grad():
    restored_logits = classify_batch(
        finetune_valid_batch["input_ids"],
        finetune_valid_batch["attention_mask"]
    )
    restored_loss = loss_fn(
        restored_logits, finetune_valid_y
    ).item()
    restored_acc = (
        restored_logits.argmax(dim=1) == finetune_valid_y
    ).float().mean().item()

print("本次最佳轮数：", best_epoch)
print(f"记录的最佳验证损失：{best_valid_loss:.4f}")
print(f"恢复后验证损失：{restored_loss:.4f}")
print(f"恢复后验证准确率：{restored_acc:.3f}")
from pathlib import Path

save_path = Path(__file__).resolve().parent / "bert_sentiment_best.pt"

checkpoint = {
    # 当前模型已经恢复为最佳参数
    "bert_state": bert.state_dict(),
    "classifier_state": classifier.state_dict(),

    # 重新加载时需要的信息
    "model_name": "google-bert/bert-base-chinese",
    "class_names": ["负面", "中性", "正面"],
    "hidden_size": bert.config.hidden_size,
    "max_length": 32,
    "pooling": "masked_mean",

    # 本次实验记录
    "best_epoch": best_epoch,
    "best_valid_loss": best_valid_loss
}

torch.save(checkpoint, save_path)

print("已保存：", save_path)
print(f"文件大小：{save_path.stat().st_size / 1024**2:.1f} MB")