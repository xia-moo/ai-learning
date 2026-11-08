import hashlib
from pathlib import Path

model_path = (
    Path(__file__).resolve().parent / "bert_sentiment_best.pt"
)

hasher = hashlib.sha256()

with model_path.open("rb") as file:
    while True:
        chunk = file.read(1024 * 1024)  # 每次读取 1 MiB

        if not chunk:
            break

        hasher.update(chunk)

print("模型文件：", model_path.name)
print("SHA-256：", hasher.hexdigest())
