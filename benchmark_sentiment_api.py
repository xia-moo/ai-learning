import json
import math
from statistics import mean, median
from time import perf_counter
from urllib.request import Request, urlopen

URL = "http://127.0.0.1:8000/predict"
BODY = json.dumps(
    {"text": "这个产品很好用"},
    ensure_ascii=False
).encode("utf-8")


def measure_once():
    request = Request(
        URL,
        data=BODY,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = perf_counter()

    with urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))

    elapsed_ms = (perf_counter() - start) * 1000

    # 确认返回的是预测结果
    assert "label" in result
    return elapsed_ms


# 预热结果不计入统计
for _ in range(3):
    measure_once()

latencies = []

for i in range(50):
    latencies.append(measure_once())

    if (i + 1) % 10 == 0:
        print(f"已完成 {i + 1}/50 次")

ordered = sorted(latencies)

# P95 使用最近秩法：取排序后第 ceil(0.95 × N) 个值
p95_rank = math.ceil(0.95 * len(ordered))
p95 = ordered[p95_rank - 1]

print(f"\n请求数量：{len(latencies)}")
print(f"平均耗时：{mean(latencies):.1f} 毫秒")
print(f"P50：{median(latencies):.1f} 毫秒")
print(f"P95：{p95:.1f} 毫秒")
print(f"最大耗时：{max(latencies):.1f} 毫秒")
from concurrent.futures import ThreadPoolExecutor, as_completed


def benchmark_concurrency(workers, request_count=50):
    durations = []
    errors = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        start = perf_counter()

        futures = [
            executor.submit(measure_once)
            for _ in range(request_count)
        ]

        for future in as_completed(futures):
            try:
                durations.append(future.result())
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")

        total_seconds = perf_counter() - start

    print(f"\n并发上限：{workers}")
    print(f"成功：{len(durations)} | 失败：{len(errors)}")
    print(f"总耗时：{total_seconds:.2f} 秒")
    print(f"成功吞吐量：{len(durations) / total_seconds:.2f} 请求/秒")

    if durations:
        ordered = sorted(durations)
        p95_index = math.ceil(0.95 * len(ordered)) - 1

        print(f"P50：{median(durations):.1f} 毫秒")
        print(f"P95：{ordered[p95_index]:.1f} 毫秒")

    if errors:
        print("首个错误：", errors[0])


for workers in [1, 2, 4]:
    benchmark_concurrency(workers)