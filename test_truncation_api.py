import json
from urllib.request import Request, urlopen

# 构造明显超过当前 64 token 上限的文本
text = "这个产品很好用。" * 30

request = Request(
    url="http://127.0.0.1:8000/predict",
    data=json.dumps(
        {"text": text}, ensure_ascii=False
    ).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urlopen(request, timeout=30) as response:
    assert response.status == 200
    result = json.loads(response.read().decode("utf-8"))

assert result["truncated"] is True, "长文本没有标记为截断"
assert (
    result["original_token_count"] > result["input_token_count"]
), "截断前后的长度不符合预期"
assert result["action"] == "转人工审核", "截断后没有转人工"
assert result["reason"] == "输入发生截断", "处理原因不正确"

print("原始 token 数：", result["original_token_count"])
print("实际输入 token 数：", result["input_token_count"])
print("处理动作：", result["action"])
print("通过：API 对截断文本执行了转人工规则")
texts = [
    "我对这次购物很满意",
    "这个产品很好用。" * 30
]

batch_request = Request(
    url="http://127.0.0.1:8000/predict/batch",
    data=json.dumps(
        {"texts": texts}, ensure_ascii=False
    ).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urlopen(batch_request, timeout=30) as response:
    assert response.status == 200
    batch_result = json.loads(response.read().decode("utf-8"))

results = batch_result["results"]

assert batch_result["count"] == 2
assert len(results) == 2
assert [item["text"] for item in results] == texts

short, long = results

assert short["truncated"] is False
assert long["truncated"] is True
assert long["action"] == "转人工审核"
assert long["reason"] == "输入发生截断"

# 短文本按本次实际概率判断，避免把模型输出写死
expected_short_action = (
    "自动通过"
    if short["probabilities"]["正面"] >= short["decision_threshold"]
    else "转人工审核"
)
assert short["action"] == expected_short_action

for item in results:
    print(
        f"有效长度={item['input_token_count']} | "
        f"截断={item['truncated']} | "
        f"动作={item['action']}"
    )

print("通过：批量接口分别判断每条文本")
total = len(results)

auto_count = sum(
    item["action"] == "自动通过"
    for item in results
)
manual_count = sum(
    item["action"] == "转人工审核"
    for item in results
)
truncated_count = sum(
    item["truncated"]
    for item in results
)

if total > 0:
    print(f"本批文本数：{total}")
    print(f"自动通过率：{auto_count / total:.1%}")
    print(f"转人工率：{manual_count / total:.1%}")
    print(f"截断率：{truncated_count / total:.1%}")
truncation_review_count = sum(
    item["action"] == "转人工审核"
    and item["reason"] == "输入发生截断"
    for item in results
)

if manual_count > 0:
    share = truncation_review_count / manual_count
    print(f"转人工样本中，截断原因占比：{share:.1%}")
else:
    print("本批没有转人工样本，原因占比无法计算")