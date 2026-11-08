import json
from statistics import mean
from time import perf_counter
from urllib.request import Request, urlopen
import math

URL = "http://127.0.0.1:8000/predict/batch"
TOTAL_TEXTS = 128
TEXT = "这个产品很好用"


def send_batch(texts):
    body = json.dumps(
        {"texts": texts},
        ensure_ascii=False
    ).encode("utf-8")

    request = Request(
        URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = perf_counter()

    with urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))

    elapsed_ms = (perf_counter() - start) * 1000

    assert result["count"] == len(texts)
    assert len(result["results"]) == len(texts)

    return elapsed_ms


for batch_size in [1, 4, 8]:
    texts = [TEXT] * batch_size
    request_count = 50
    text_count = request_count * batch_size

    # 每种批大小分别预热
    for _ in range(3):
        send_batch(texts)

    start = perf_counter()

    latencies = [
        send_batch(texts)
        for _ in range(request_count)
    ]

    total_seconds = perf_counter() - start

    print(f"\n每批文本数：{batch_size}")
    print(f"请求数：{request_count}")
    print(f"处理文本数：{text_count}")
    print(f"总耗时：{total_seconds:.2f} 秒")
    print(f"平均每批请求耗时：{mean(latencies):.1f} 毫秒")
    print(f"请求吞吐量：{request_count / total_seconds:.2f} 请求/秒")
    print(f"文本吞吐量：{text_count / total_seconds:.2f} 条/秒")
    ordered = sorted(latencies)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]

    over_limit = sum(ms > 100 for ms in latencies)

    print(f"P95：{p95:.1f} 毫秒")
    print(
        f"超过 100 毫秒：{over_limit}/{len(latencies)}"
        f"（{over_limit / len(latencies):.1%}）"
    )