import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from time import perf_counter

def send_request(payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = Request(
        "http://127.0.0.1:8000/predict",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    start = perf_counter()
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.loads(
                response.read().decode("utf-8")
            )

    except HTTPError as error:
        with error:
            return error.code, json.loads(
                error.read().decode("utf-8")
            )


cases = [
    ("正常文本", {"text": "这个产品很好用"}, 200),
    ("空白文本", {"text": "   "}, 400),
    ("缺少字段", {}, 422)
]

for name, payload, expected_status in cases:
    status, result = send_request(payload)

    assert status == expected_status, (
        f"{name}：预期 {expected_status}，实际 {status}"
    )

    if status == 200:
        assert result["label"] in ["负面", "中性", "正面"]
        assert set(result["probabilities"]) == {"负面", "中性", "正面"}

    print(f"通过：{name}，状态码 {status}")

print("全部接口检查通过")