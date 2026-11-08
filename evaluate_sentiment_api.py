import json
from urllib.request import Request, urlopen
from pathlib import Path

# 这些例子之前已经使用过，只用于开发检查，不是独立测试集
data_path = (
    Path(__file__).resolve().parent / "sentiment_dev.jsonl"
)

examples = []

allowed_labels = {"正面", "负面", "中性"}

with data_path.open("r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"第 {line_number} 行 JSON 格式错误：{error.msg}"
            ) from error

        if not isinstance(item, dict):
            raise ValueError(f"第 {line_number} 行必须是 JSON 对象")

        text = item.get("text")
        label = item.get("label")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"第 {line_number} 行 text 必须是非空字符串")

        if not isinstance(label, str) or label not in allowed_labels:
            raise ValueError(
                f"第 {line_number} 行 label 必须是正面、负面或中性"
            )

        examples.append({
            "text": text.strip(),
            "label": label
        })

if not examples:
    raise ValueError("评估数据不能为空")

texts = [item["text"] for item in examples]
seen = {}

for index, example in enumerate(examples, start=1):
    text = example["text"].strip()
    label = example["label"]

    if text in seen:
        previous_index, previous_label = seen[text]

        if label != previous_label:
            raise ValueError(
                f"标签冲突：第 {previous_index} 条与第 {index} 条"
                f"文本相同，标签却不同"
            )

        raise ValueError(
            f"重复文本：第 {previous_index} 条与第 {index} 条相同"
        )

    seen[text] = (index, label)

print(f"重复检查通过：{len(seen)} 条不同文本")

request = Request(
    url="http://127.0.0.1:8000/predict/batch",
    data=json.dumps(
        {"texts": texts}, ensure_ascii=False
    ).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urlopen(request, timeout=30) as response:
    data = json.loads(response.read().decode("utf-8"))

results = data["results"]

# 先检查数量和顺序，避免把预测与错误的标签配对
assert len(results) == len(examples), "返回数量不一致"
assert [item["text"] for item in results] == texts, "返回顺序不一致"

correct_count = 0

for example, result in zip(examples, results):
    correct = result["label"] == example["label"]
    correct_count += int(correct)

    print(
        f"{example['text']} | "
        f"真实={example['label']} | "
        f"预测={result['label']} | "
        f"{'正确' if correct else '错误'}"
    )

print(f"\n正确数量：{correct_count}/{len(examples)}")
print(f"开发样例准确率：{correct_count / len(examples):.1%}")
passed_count = 0
passed_positive_count = 0

for example, result in zip(examples, results):
    if result["action"] == "自动通过":
        passed_count += 1

        if example["label"] == "正面":
            passed_positive_count += 1

print(f"自动通过数量：{passed_count}/{len(examples)}")
print(f"自动通过率：{passed_count / len(examples):.1%}")

if passed_count > 0:
    precision = passed_positive_count / passed_count
    print(f"通过样本的精准率：{precision:.1%}")
else:
    print("通过样本的精准率：无法计算")
positive_count = sum(
    example["label"] == "正面"
    for example in examples
)

if positive_count > 0:
    positive_pass_recall = passed_positive_count / positive_count
    print(f"正面自动通过召回率：{positive_pass_recall:.1%}")
else:
    print("正面自动通过召回率：无法计算")
# 把“自动通过”当作预测正类，把“真实正面”当作真实正类
tp = passed_positive_count
fp = passed_count - tp
fn = positive_count - tp

denominator = 2 * tp + fp + fn

if denominator > 0:
    policy_f1 = 2 * tp / denominator
    print(f"自动通过规则 F1：{policy_f1:.1%}")
else:
    print("自动通过规则 F1：无法计算")
tn = len(examples) - tp - fp - fn

print("\n自动通过规则混淆矩阵：")
print(f"真实正面   | 自动通过={tp} | 转人工={fn}")
print(f"真实非正面 | 自动通过={fp} | 转人工={tn}")
print("\n规则误放行与未自动通过的正面样例：")

for example, result in zip(examples, results):
    actual_positive = example["label"] == "正面"
    auto_passed = result["action"] == "自动通过"

    if auto_passed and not actual_positive:
        print(f"FP｜误放行：{example['text']}")

    elif actual_positive and not auto_passed:
        print(
            f"FN｜真实正面未自动通过：{example['text']} | "
            f"原因：{result['reason']}"
        )
from sentiment_policy import add_decision

print("\n开发样例上的阈值对比：")
cost_results = []
for threshold in [0.80, 0.85, 0.90]:
    passed = 0
    true_positive = 0

    for example, prediction in zip(examples, results):
        decision = add_decision(prediction, threshold)

        if decision["action"] == "自动通过":
            passed += 1
            if example["label"] == "正面":
                true_positive += 1
                
    false_passed = passed - true_positive
    manual_count = len(examples) - passed

    total_cost = false_passed * 10 + manual_count * 1
    precision = true_positive / passed if passed else None

    cost_results.append({
        "threshold": threshold,
        "total_cost": total_cost,
        "false_passed": false_passed,
        "manual_count": manual_count,
        "precision": precision,
        "passed_count": passed,
    })
    recall = (
        true_positive / positive_count
        if positive_count else None
    )

    precision_text = (
        f"{precision:.1%}" if precision is not None else "无法计算"
    )
    recall_text = (
        f"{recall:.1%}" if recall is not None else "无法计算"
    )

    print(
        f"阈值={threshold:.2f} | "
        f"自动通过率={passed / len(examples):.1%} | "
        f"精准率={precision_text} | "
        f"召回率={recall_text} | "
        f"误放行={false_passed} | "
        f"转人工={manual_count} | "
        f"总代价={total_cost}"
    )
required_precision = 0.95
minimum_passed_count = 3  # 仅用于演示筛选逻辑

eligible = [
    item
    for item in cost_results
    if item["precision"] is not None
    and item["precision"] >= required_precision
    and item["passed_count"] >= minimum_passed_count
]

if eligible:
    best_by_cost = min(
        eligible,
        key=lambda item: (
            item["total_cost"],
            item["false_passed"],
            -item["threshold"]
        )
    )

    print(f"\n达标候选中的最低代价阈值：{best_by_cost['threshold']:.2f}")
    print(f"精准率：{best_by_cost['precision']:.1%}")
    print("总代价：", best_by_cost["total_cost"])
else:
    print("\n没有同时满足精准率和最少通过样本数要求的候选阈值")
print("\n否定表达对照：")

for example, result in zip(examples, results):
    if "失望" in example["text"]:
        print(
            f"\n{example['text']} | "
            f"真实={example['label']} | 预测={result['label']}"
        )

        for label, probability in result["probabilities"].items():
            print(f"{label}：{probability:.3f}")
baseline = {
    "description": "否定表达补充训练前的开发集结果",
    "samples": [
        {
            "text": example["text"],
            "true_label": example["label"],
            "prediction": prediction
        }
        for example, prediction in zip(examples, results)
    ]
}

baseline_path = (
    Path(__file__).resolve().parent / "sentiment_baseline_before.json"
)

# x 表示仅在文件不存在时创建，防止后续运行覆盖训练前结果
try:
    with baseline_path.open("x", encoding="utf-8") as file:
        json.dump(baseline, file, ensure_ascii=False, indent=2)
except FileExistsError:
    print("训练前基准已存在，保留原文件：", baseline_path)
else:
    print("训练前基准已保存：", baseline_path)
    
report = {
    "dataset_role": "development",
    "samples": [
        {
            "text": example["text"],
            "true_label": example["label"],
            "prediction": prediction
        }
        for example, prediction in zip(examples, results)
    ]
}

report_path = (
    Path(__file__).resolve().parent / "sentiment_before_dev10.json"
)

# 只创建一次，避免之后切回旧模型时覆盖新模型结果
try:
    with report_path.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
except FileExistsError:
    print("报告已存在，本次结果未写入，保留原文件：", report_path)
else:
    print("报告已保存：", report_path)