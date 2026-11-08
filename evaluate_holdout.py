import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

project_dir = Path(__file__).resolve().parent
data_path = project_dir / "sentiment_holdout.jsonl"
report_path = project_dir / "sentiment_holdout_report.json"

# 防止重复运行时覆盖首次评估结果
if report_path.exists():
    raise FileExistsError(f"报告已存在，请保留首次结果：{report_path}")

data_bytes = data_path.read_bytes()
examples = [
    json.loads(line)
    for line in data_bytes.decode("utf-8").splitlines()
    if line.strip()
]

allowed_labels = {"负面", "中性", "正面"}

if not examples:
    raise ValueError("评估数据不能为空")

for number, item in enumerate(examples, start=1):
    if not isinstance(item, dict):
        raise ValueError(f"第 {number} 条必须是对象")

    if (
        not isinstance(item.get("text"), str)
        or not item["text"].strip()
    ):
        raise ValueError(f"第 {number} 条文本不合法")

    if (
        not isinstance(item.get("label"), str)
        or item["label"] not in allowed_labels
    ):
        raise ValueError(f"第 {number} 条标签不合法")

texts = [item["text"].strip() for item in examples]
if len(set(texts)) != len(texts):
    raise ValueError("评估数据包含重复文本")

# 计算预期检查点的文件指纹
hasher = hashlib.sha256()
with (project_dir / "bert_sentiment_negation_v1.pt").open("rb") as file:
    while chunk := file.read(1024 * 1024):
        hasher.update(chunk)

expected_hash = hasher.hexdigest()

request = Request(
    url="http://127.0.0.1:8000/predict/batch",
    data=json.dumps(
        {"texts": texts}, ensure_ascii=False
    ).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urlopen(request, timeout=60) as response:
    results = json.loads(response.read().decode("utf-8"))["results"]

if [item["text"] for item in results] != texts:
    raise ValueError("返回文本数量或顺序不一致")

# 确认实际服务配置，不依赖手工记录
for item in results:
    if (
        item["checkpoint_sha256"] != expected_hash
        or item["inference_mode"] != "FP32"
        or item["max_length"] != 64
        or item["decision_threshold"] != 0.90
    ):
        raise ValueError("服务配置与本次固定配置不一致，请检查并重启 API")

correct = 0
passed = 0
true_passed = 0

for example, result in zip(examples, results):
    is_correct = result["label"] == example["label"]
    correct += int(is_correct)

    if result["action"] == "自动通过":
        passed += 1
        true_passed += int(example["label"] == "正面")

    print(
        f"{'正确' if is_correct else '错误'} | "
        f"{result['text']} | "
        f"真实={example['label']} | "
        f"预测={result['label']} | "
        f"{result['action']}"
    )

total = len(examples)
positive_count = sum(item["label"] == "正面" for item in examples)

metrics = {
    "total": total,
    "correct": correct,
    "accuracy": correct / total,
    "auto_count": passed,
    "auto_rate": passed / total,
    "auto_precision": true_passed / passed if passed else None,
    "positive_auto_recall": (
        true_passed / positive_count if positive_count else None
    ),
    "false_passed": passed - true_passed,
    "truncated_count": sum(item["truncated"] for item in results)
}

report = {
    "dataset_description": "人工编写的新增检查样例，固定配置评估",
    "dataset_sha256": hashlib.sha256(data_bytes).hexdigest(),
    "checkpoint_sha256": expected_hash,
    "metrics": metrics,
    "samples": [
        {
            "text": result["text"],
            "true_label": example["label"],
            "prediction": result
        }
        for example, result in zip(examples, results)
    ]
}

with report_path.open("x", encoding="utf-8") as file:
    json.dump(report, file, ensure_ascii=False, indent=2)

print("\n评估指标：")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print("报告已保存：", report_path)