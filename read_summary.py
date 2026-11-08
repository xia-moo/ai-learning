import json
from pathlib import Path

summary_path = Path(__file__).resolve().parent / "batch_summary.json"

summary = json.loads(
    summary_path.read_text(encoding="utf-8")
)

print("记录时间：", summary["recorded_at"])
print("文本总数：", summary["total"])
print("自动通过数量：", summary["auto_count"])
print("转人工数量：", summary["manual_count"])

auto_rate = summary["auto_rate"]
if auto_rate is not None:
    print(f"自动通过率：{auto_rate:.1%}")
else:
    print("没有文本，自动通过率无法计算")
print("记录的阈值：", summary.get("decision_thresholds", "未记录"))
print("记录的长度上限：", summary.get("inference_max_lengths", "未记录"))
print("记录的推理模式：", summary.get("inference_modes", "未记录"))
print(
    "记录的检查点哈希：",
    summary.get("checkpoint_sha256s", "未记录")
)