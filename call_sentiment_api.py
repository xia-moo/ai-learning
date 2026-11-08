import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from time import perf_counter

payload = {
    "text": (
        "这款产品有多种颜色，包装里附带说明书和充电线，"
        "外观与页面描述一致，但是用了两天就坏了，我非常失望。"
    )
}

# Python 字典 → JSON 字符串 → UTF-8 字节
body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

request = Request(
    url="http://127.0.0.1:8000/predict",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST"
)
start = perf_counter()
try:
    with urlopen(request, timeout=30) as response:
        status_code = response.status
        result = json.loads(response.read().decode("utf-8"))

except HTTPError as error:
    # 已经收到服务器响应，但请求未成功
    print("HTTP 错误状态码：", error.code)
    print("错误详情：", error.read().decode("utf-8"))

except URLError as error:
    # 例如服务没启动，无法建立连接
    print("连接失败：", error.reason)

except TimeoutError:
    print("请求超时")

else:
    # 只有请求成功才读取预测结果
    elapsed_ms = (perf_counter() - start) * 1000
    if result["truncated"]:
        print(
            "提示：输入发生截断。"
            f"完整文本有 {result['original_token_count']} 个 token，"
            f"模型实际接收 {result['input_token_count']} 个。"
        )
    print("本次预测只基于保留的内容，可能遗漏后文信息。")
    print(f"本次请求耗时：{elapsed_ms:.1f} 毫秒")
    print("状态码：", status_code)
    print("输入文本：", result["text"])
    print("预测类别：", result["label"])
    print("当前长度上限：", result["max_length"])
    print("完整 token 数：", result["original_token_count"])
    print("实际输入 token 数：", result["input_token_count"])
    print("是否截断：", result["truncated"])

    for name, probability in result["probabilities"].items():
        print(f"{name}：{probability:.3f}")