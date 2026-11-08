import torch
from torch import nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)

# 1. 读取并划分数据
X, y = load_iris(return_X_y=True)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. 只用训练集确定标准化规则
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# 3. 转换成 PyTorch 张量
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_valid_t = torch.tensor(X_valid, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_valid_t = torch.tensor(y_valid, dtype=torch.long)

# 4. 建立网络：4 个输入 → 8 个隐藏单元 → 3 个类别分数
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 3)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# 5. 训练 300 轮
for epoch in range(300):
    model.train()
    optimizer.zero_grad()

    logits = model(X_train_t)
    loss = loss_fn(logits, y_train_t)

    loss.backward()
    optimizer.step()

    # 每 50 轮检查更新后的表现
    if (epoch + 1) % 50 == 0:
        model.eval()

        with torch.no_grad():
            train_loss = loss_fn(model(X_train_t), y_train_t)

            valid_logits = model(X_valid_t)
            valid_loss = loss_fn(valid_logits, y_valid_t)

            predicted = valid_logits.argmax(dim=1)
            valid_acc = (predicted == y_valid_t).float().mean()

        print(
            f"第 {epoch + 1} 轮 | "
            f"训练损失={train_loss.item():.4f} | "
            f"验证损失={valid_loss.item():.4f} | "
            f"验证准确率={valid_acc.item():.3f}"
        )
        from torch.utils.data import TensorDataset, DataLoader

# 把每个样本的特征与标签配成一对
train_dataset = TensorDataset(X_train_t, y_train_t)

# 每批 20 条，每次遍历时打乱样本顺序
train_loader = DataLoader(
    train_dataset,
    batch_size=20,
    shuffle=True
)

# 查看每一批的形状
for batch_index, (batch_X, batch_y) in enumerate(train_loader, start=1):
    print(
        f"第 {batch_index} 批 | "
        f"特征：{tuple(batch_X.shape)} | "
        f"标签：{tuple(batch_y.shape)}"
    )
    
    
# 从新模型开始训练
from copy import deepcopy

best_loss = float("inf")
best_state = None
best_epoch = 0

patience = 20
wait = 0

for epoch in range(300):
    model.train()

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        logits = model(batch_X)
        loss = loss_fn(logits, batch_y)

        loss.backward()
        optimizer.step()

    # 每轮都检查验证损失
    model.eval()
    with torch.no_grad():
        valid_loss = loss_fn(model(X_valid_t), y_valid_t).item()

    if valid_loss < best_loss:
        # 有改善：保留当前参数，等待计数归零
        best_loss = valid_loss
        best_state = deepcopy(model.state_dict())
        best_epoch = epoch + 1
        wait = 0
    else:
        # 没有改善：等待计数加一
        wait += 1

    if (epoch + 1) % 20 == 0:
        print(
            f"第 {epoch + 1} 轮 | "
            f"验证损失={valid_loss:.4f} | "
            f"连续未改善={wait}"
        )

    if wait >= patience:
        print(f"触发早停：第 {epoch + 1} 轮")
        break

# 恢复验证损失最低时的参数
model.load_state_dict(best_state)
from pathlib import Path

iris_info = load_iris()

checkpoint = {
    "model_state": model.state_dict(),
    "layer_sizes": [4, 8, 3],

    # 保存训练时确定的标准化规则
    "scaler_mean": torch.tensor(
        scaler.mean_, dtype=torch.float64
    ),
    "scaler_scale": torch.tensor(
        scaler.scale_, dtype=torch.float64
    ),

    "feature_names": list(iris_info.feature_names),
    "class_names": iris_info.target_names.tolist(),

    "best_epoch": best_epoch,
    "best_valid_loss": best_loss
}

# 保存在当前 Python 文件旁边
save_path = Path(__file__).resolve().parent / "iris_checkpoint.pt"
torch.save(checkpoint, save_path)

print("完整检查点已保存：", save_path)
from pathlib import Path
import torch
from torch import nn

# 1. 加载检查点
checkpoint_path = Path(__file__).resolve().parent / "iris_checkpoint.pt"

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

# 2. 按保存的信息重建网络结构
input_size, hidden_size, output_size = checkpoint["layer_sizes"]

model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
)

# 3. 恢复训练好的参数，切换到评估模式
model.load_state_dict(checkpoint["model_state"])
model.eval()

# 4. 新样本：花萼长度、花萼宽度、花瓣长度、花瓣宽度
raw_features = torch.tensor(
    [[5.1, 3.5, 1.4, 0.2]],
    dtype=torch.float64
)

# 使用训练时保存的标准化规则
scaled_features = (
    raw_features - checkpoint["scaler_mean"]
) / checkpoint["scaler_scale"]

# 5. 预测
with torch.no_grad():
    logits = model(scaled_features.float())
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = probabilities.argmax(dim=1).item()

class_names = checkpoint["class_names"]

print("输入特征顺序：", checkpoint["feature_names"])
print("预测品种：", class_names[predicted_class])

for name, probability in zip(class_names, probabilities[0].tolist()):
    print(f"{name}：{probability:.4f}")