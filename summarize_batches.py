import json
from urllib.request import Request, urlopen


def call_batch_api(texts):
    request = Request(
        url="http://127.0.0.1:8000/predict/batch",
        data=json.dumps(
            {"texts": texts}, ensure_ascii=False
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


text_batches = [
    ["好用", "我对这次购物很满意"],
    ["这个产品很好用。" * 30],
]

total = 0
auto_count = 0
manual_count = 0
truncated_count = 0

observed_thresholds = set()
observed_max_lengths = set()
observed_inference_modes = set()
observed_checkpoint_hashes = set()
for batch_number, texts in enumerate(text_batches, start=1):
    response = call_batch_api(texts)
    results = response["results"]
    observed_checkpoint_hashes.update(
    item["checkpoint_sha256"] for item in results
    )
    observed_inference_modes.update(
    item["inference_mode"] for item in results
    )
    observed_max_lengths.update(
    item["max_length"] for item in results
    )
    observed_thresholds.update(
    item["decision_threshold"] for item in results
    )

    total += len(results)
    auto_count += sum(
        item["action"] == "自动通过" for item in results
    )
    manual_count += sum(
        item["action"] == "转人工审核" for item in results
    )
    truncated_count += sum(
        item["truncated"] for item in results
    )

    print(
        f"第 {batch_number} 批完成 | "
        f"累计文本={total} | "
        f"累计自动通过={auto_count} | "
        f"累计转人工={manual_count}"
    )

if total > 0:
    print(f"\n总体自动通过率：{auto_count / total:.1%}")
    print(f"总体转人工率：{manual_count / total:.1%}")
    print(f"总体截断率：{truncated_count / total:.1%}")

from pathlib import Path
from datetime import datetime

summary = {
    "recorded_at": datetime.now().astimezone().isoformat(
        timespec="seconds"
    ),
    "total": total,
    "auto_count": auto_count,
    "manual_count": manual_count,
    "truncated_count": truncated_count,
    "auto_rate": auto_count / total if total else None,
    "manual_rate": manual_count / total if total else None,
    "truncated_rate": truncated_count / total if total else None,
    "decision_thresholds": sorted(observed_thresholds),
    "inference_max_lengths": sorted(observed_max_lengths),
    "inference_modes": sorted(observed_inference_modes),
    "checkpoint_sha256s": sorted(observed_checkpoint_hashes),
}

output_path = Path(__file__).resolve().parent / "batch_summary.json"

output_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("统计已保存：", output_path)
history_path = (
    Path(__file__).resolve().parent / "batch_summary_history.jsonl"
)

with history_path.open("a", encoding="utf-8") as file:
    file.write(
        json.dumps(summary, ensure_ascii=False) + "\n"
    )

print("历史记录已追加：", history_path)