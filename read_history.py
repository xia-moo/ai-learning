import json
from pathlib import Path

history_path = (
    Path(__file__).resolve().parent / "batch_summary_history.jsonl"
)

records = []

with history_path.open("r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            records.append(json.loads(line))

print("历史运行次数：", len(records))

for number, record in enumerate(records, start=1):
    rate = record["auto_rate"]
    rate_text = f"{rate:.1%}" if rate is not None else "无法计算"

    thresholds = record.get("decision_thresholds")

    if thresholds is None:
        threshold_text = "未记录"
    elif not thresholds:
        threshold_text = "未观测到阈值"
    else:
        threshold_text = ", ".join(
            f"{value:.2f}" for value in thresholds
        )

    print(
        f"第 {number} 次 | "
        f"时间={record['recorded_at']} | "
        f"文本数={record['total']} | "
        f"自动通过率={rate_text} | "
        f"阈值={threshold_text}"
    )
total_processed = sum(record["total"] for record in records)
total_auto = sum(record["auto_count"] for record in records)
total_manual = sum(record["manual_count"] for record in records)

print("\n历史累计处理条次：", total_processed)
print("历史累计自动通过条次：", total_auto)
print("历史累计转人工条次：", total_manual)

if total_processed > 0:
    print(f"历史总体自动通过率：{total_auto / total_processed:.1%}")
else:
    print("没有处理记录，无法计算自动通过率")
    
target_threshold = 0.90
target_max_length = 64
target_inference_mode = "FP32"
target_checkpoint_sha256 = (
    "c53e21798770b7a9be3a707795337da7e604fb10eb8000a0b36466bd4a9f564d"
)

matched_records = []

for record in records:
    thresholds = record.get("decision_thresholds")

    max_lengths = record.get("inference_max_lengths")
    if max_lengths is None:
        max_lengths = record.get("max_lengths")

    inference_modes = record.get("inference_modes")

    if (
        thresholds == [target_threshold]
        and max_lengths == [target_max_length]
        and inference_modes == [target_inference_mode]
        and record.get("checkpoint_sha256s")
        == [target_checkpoint_sha256]
    ):
        matched_records.append(record)

matched_total = sum(
    record["total"] for record in matched_records
)
matched_auto = sum(
    record["auto_count"] for record in matched_records
)

print(
    f"\n仅统计阈值 {target_threshold:.2f}、"
    f"长度上限 {target_max_length}、"
    f"模式 {target_inference_mode}："
)
print("检查点 SHA-256：", target_checkpoint_sha256)
print("匹配运行次数：", len(matched_records))
print("累计处理条次：", matched_total)

if matched_total > 0:
    print(f"自动通过率：{matched_auto / matched_total:.1%}")
else:
    print("没有符合条件的处理记录")
