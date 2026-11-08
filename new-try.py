from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. 加载数据，保留表格形式
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

print("特征前五行：")
print(X.head())
print("类别名称：", iris.target_names)

# 2. 划分训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. 训练决策树
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 4. 分别检查训练集和验证集表现
train_pred = model.predict(X_train)
valid_pred = model.predict(X_valid)

print("训练集形状：", X_train.shape)
print("验证集形状：", X_valid.shape)
print("训练集准确率：", accuracy_score(y_train, train_pred))
print("验证集准确率：", accuracy_score(y_valid, valid_pred))
for depth in [1, 2, 3, 4, 5, None]:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    train_acc = accuracy_score(
        y_train, model.predict(X_train)
    )
    valid_acc = accuracy_score(
        y_valid, model.predict(X_valid)
    )

    print(
        f"最大深度：{str(depth):>4} | "
        f"训练准确率：{train_acc:.3f} | "
        f"验证准确率：{valid_acc:.3f}"
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_score

# 指定如何划分 5 折
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 指定要评估的模型
model = DecisionTreeClassifier(max_depth=3, random_state=42)

# 自动完成每一折的训练和验证
scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print("每折准确率：", scores)
print("平均准确率：", scores.mean())
print("准确率标准差：", scores.std())
for depth in [1, 2, 3, 4, 5, None]:
    candidate = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    scores = cross_val_score(
        candidate,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )
    
final_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
final_model.fit(X_train, y_train)
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest_scores = cross_val_score(
    forest,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print("随机森林每折准确率：", forest_scores)
print("随机森林平均准确率：", forest_scores.mean())
print("随机森林标准差：", forest_scores.std())
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 把标准化和模型组成一个完整流程
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_scores = cross_val_score(
    logistic_model,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print("逻辑回归每折准确率：", logistic_scores)
print("逻辑回归平均准确率：", logistic_scores.mean())
print("逻辑回归标准差：", logistic_scores.std())
from sklearn.metrics import confusion_matrix, classification_report

# 交叉验证不会把原来的 logistic_model 训练好
# 这里用全部 120 条训练数据训练完整流程
logistic_model.fit(X_train, y_train)

# 对原来的 30 条验证数据预测
valid_pred = logistic_model.predict(X_valid)

print("类别顺序：", iris.target_names)

print("混淆矩阵：")
print(confusion_matrix(
    y_valid,
    valid_pred,
    labels=[0, 1, 2]
))

print("分类报告：")
print(classification_report(
    y_valid,
    valid_pred,
    labels=[0, 1, 2],
    target_names=iris.target_names,
    digits=3
))