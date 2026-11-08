import json
from pathlib import Path
from collections import Counter

project_dir = Path(__file__).resolve().parent


def read_samples(filename):
    path = project_dir / filename
    report = json.loads(path.read_text(encoding="utf-8"))
    return report["samples"]


before = read_samples("sentiment_before_dev10.json")
after = read_samples("sentiment_after_dev10.json")

# 先确认比较的是相同文本、相同标签、相同顺序
before_data = [(item["text"], item["true_label"]) for item in before]
after_data = [(item["text"], item["true_label"]) for item in after]

if not before_data or before_data != after_data:
    raise ValueError("评估数据为空，或两份报告的文本、标签、顺序不一致")

counts = Counter()

for old, new in zip(before, after):
    # 模型不同，其他比较条件应保持一致
    for key in ["inference_mode", "max_length", "decision_threshold"]:
        if old["prediction"][key] != new["prediction"][key]:
            raise ValueError(f"比较配置不一致：{key}")

    label = old["true_label"]
    old_correct = old["prediction"]["label"] == label
    new_correct = new["prediction"]["label"] == label

    if not old_correct and new_correct:
        category = "改善"
    elif old_correct and not new_correct:
        category = "退步"
    elif old_correct and new_correct:
        category = "保持正确"
    else:
        category = "仍然错误"

    counts[category] += 1

    if category in ["改善", "退步"]:
        print(f"{category}：{old['text']}")

print()
for category in ["改善", "退步", "保持正确", "仍然错误"]:
    print(f"{category}：{counts[category]} 条")
print("\n处理动作变化：")
action_changes = 0

for old, new in zip(before, after):
    old_action = old["prediction"]["action"]
    new_action = new["prediction"]["action"]

    if old_action != new_action:
        action_changes += 1
        print(
            f"{old['text']} | "
            f"真实={old['true_label']} | "
            f"{old_action} → {new_action}"
        )

print(f"动作变化数量：{action_changes}")
print("\n处理动作变化的影响：")

for old, new in zip(before, after):
    old_passed = old["prediction"]["action"] == "自动通过"
    new_passed = new["prediction"]["action"] == "自动通过"
    is_positive = old["true_label"] == "正面"

    if old_passed == new_passed:
        continue

    if new_passed:
        impact = "真实正面新增自动通过" if is_positive else "新增误放行"
    else:
        impact = "真实正面新增转人工" if is_positive else "消除误放行"

    print(f"{impact}：{old['text']}")