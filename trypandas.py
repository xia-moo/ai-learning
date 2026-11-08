import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.DataFrame({
    "study_hours": [1, 2, 1, 3, 2, 4, 5, 6, 7, 8],
    "attendance": [50, 60, 55, 65, 70, 80, 85, 90, 95, 95],
    "passed": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
})

X = df[["study_hours", "attendance"]]
y = df["passed"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("训练集：", X_train.shape, y_train.shape)
print("验证集：", X_valid.shape, y_valid.shape)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. 创建模型，限制最大深度为 2
model = DecisionTreeClassifier(max_depth=2, random_state=42)

# 2. 使用训练集学习
model.fit(X_train, y_train)

# 3. 根据验证集的特征进行预测
y_pred = model.predict(X_valid)

# 4. 对比预测结果和真实答案
result = pd.DataFrame({
    "真实答案": y_valid,
    "模型预测": y_pred
})

print(result)
print("验证集准确率：", accuracy_score(y_valid, y_pred))
# 两个新学生，还不知道他们是否及格
new_students = pd.DataFrame({
    "study_hours": [2.5, 5],
    "attendance": [75, 88]
})

# 使用已经训练好的模型
predictions = model.predict(new_students)

# 展示预测结果
result = new_students.copy()
result["预测是否及格"] = predictions

print(result)