from sentiment_policy import add_decision

def evaluate_threshold(probabilities, labels, threshold):
    if len(probabilities) != len(labels):
        raise ValueError("概率数量与标签数量必须一致")
    if not labels:
        raise ValueError("评估数据不能为空")

    passed_count = 0
    correct_count = 0

    for probability, label in zip(probabilities, labels):
        if probability >= threshold:
            passed_count += 1

            if label == "正面":
                correct_count += 1

    pass_rate = passed_count / len(labels)
    precision = (
        correct_count / passed_count
        if passed_count > 0
        else None
    )

    return {
        "threshold": threshold,
        "passed_count": passed_count,
        "correct_count": correct_count,
        "pass_rate": pass_rate,
        "precision": precision,
    }


positive_probs = [0.95, 0.85, 0.82, 0.75, 0.60]
labels = ["正面", "正面", "负面", "正面", "中性"]

for threshold in [0.70, 0.80, 0.85, 0.90, 0.99]:
    result = evaluate_threshold(
        positive_probs, labels, threshold
    )

    precision_text = (
        f"{result['precision']:.1%}"
        if result["precision"] is not None
        else "无法计算"
    )

    print(
        f"阈值={threshold:.2f} | "
        f"通过={result['passed_count']}/{len(labels)} | "
        f"其中真实正面={result['correct_count']} | "
        f"自动通过率={result['pass_rate']:.1%} | "
        f"精准率={precision_text}"
    )
target_precision = 0.95
thresholds = [0.70, 0.80, 0.85, 0.90, 0.99]

results = [
    evaluate_threshold(positive_probs, labels, threshold)
    for threshold in thresholds
]

eligible = [
    result
    for result in results
    if result["precision"] is not None
    and result["precision"] >= target_precision
]

if not eligible:
    print("没有满足精准率要求的候选阈值")
else:
    best = max(
        eligible,
        key=lambda result: result["pass_rate"]
    )

    print(f"\n选中的阈值：{best['threshold']:.2f}")
    print(f"自动通过率：{best['pass_rate']:.1%}")
    print(f"精准率：{best['precision']:.1%}")
# 模拟独立测试数据，仅用于练习
test_probs = [0.93, 0.88, 0.86, 0.78, 0.55]
test_labels = ["正面", "负面", "正面", "正面", "中性"]

if eligible:
    # 沿用验证阶段选出的阈值
    test_result = evaluate_threshold(
        test_probs,
        test_labels,
        best["threshold"]
    )

    precision = test_result["precision"]
    precision_text = (
        f"{precision:.1%}" if precision is not None else "无法计算"
    )

    print(f"\n测试使用的阈值：{best['threshold']:.2f}")
    print(f"测试自动通过率：{test_result['pass_rate']:.1%}")
    print(f"测试精准率：{precision_text}")

def decide_action(positive_probability, threshold):
    if positive_probability >= threshold:
        return "自动通过"
    return "转人工审核"

example = {
    "text": "这个产品还挺好用",
    "label": "正面",
    "probabilities": {
        "负面": 0.10,
        "中性": 0.15,
        "正面": 0.75
    },
    "truncated": False
}

result = add_decision(example, threshold=0.85)

print("预测类别：", result["label"])
print("处理动作：", result["action"])
print("判断阈值：", result["decision_threshold"])

cases = [
    (0.90, False),  # 未截断，概率达标
    (0.75, False),  # 未截断，概率未达标
    (0.96, True),   # 已截断，即使概率高也转人工
]

for probability, truncated in cases:
    prediction = {
        "probabilities": {"正面": probability},
        "truncated": truncated
    }

    result = add_decision(prediction, threshold=0.85)

    print(
        f"正面概率={probability:.2f} | "
        f"截断={truncated} | "
        f"{result['action']} | "
        f"原因：{result['reason']}"
    )
def evaluate_policy(predictions, labels, threshold):
    if len(predictions) != len(labels):
        raise ValueError("预测数量与标签数量必须一致")
    if not labels:
        raise ValueError("评估数据不能为空")

    passed_count = 0
    correct_count = 0

    for prediction, label in zip(predictions, labels):
        result = add_decision(prediction, threshold)

        if result["action"] == "自动通过":
            passed_count += 1
            if label == "正面":
                correct_count += 1

    return {
        "passed_count": passed_count,
        "pass_rate": passed_count / len(labels),
        "precision": (
            correct_count / passed_count
            if passed_count > 0
            else None
        )
    }


# 模拟数据：第二条概率达标，但发生截断
demo_predictions = [
    {"probabilities": {"正面": 0.90}, "truncated": False},
    {"probabilities": {"正面": 0.96}, "truncated": True},
    {"probabilities": {"正面": 0.75}, "truncated": False},
    {"probabilities": {"正面": 0.88}, "truncated": False},
]
demo_labels = ["正面", "正面", "正面", "负面"]

metrics = evaluate_policy(
    demo_predictions, demo_labels, threshold=0.85
)

print("自动通过数量：", metrics["passed_count"])
print(f"自动通过率：{metrics['pass_rate']:.1%}")

precision = metrics["precision"]
print(
    "通过样本的精准率：",
    f"{precision:.1%}" if precision is not None else "无法计算"
)
candidates = []

for threshold in [0.80, 0.85, 0.90, 0.95]:
    metrics = evaluate_policy(
        demo_predictions,
        demo_labels,
        threshold
    )

    precision = metrics["precision"]

    if precision is not None and precision >= 0.95:
        candidates.append({
            "threshold": threshold,
            **metrics
        })

if candidates:
    best_policy = max(
        candidates,
        key=lambda item: item["pass_rate"]
    )

    print("完整规则选中的阈值：", best_policy["threshold"])
    print(f"自动通过率：{best_policy['pass_rate']:.1%}")
    print(f"精准率：{best_policy['precision']:.1%}")
else:
    print("没有满足要求的候选阈值")