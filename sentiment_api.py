from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from predict_sentiment import predict_one, predict_batch
from sentiment_policy import add_decision
from predict_sentiment import USE_INT8
from predict_sentiment import CHECKPOINT_SHA256

# 教学演示值，尚未通过真实验证集确定
DECISION_THRESHOLD = 0.90

app = FastAPI(title="中文情感分类接口")


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(request: PredictRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="输入文本不能为空"
        )

    prediction = predict_one(text)
    prediction["inference_mode"] = "INT8" if USE_INT8 else "FP32"
    prediction["checkpoint_sha256"] = CHECKPOINT_SHA256

    return add_decision(prediction, DECISION_THRESHOLD)

class BatchPredictRequest(BaseModel):
    texts: list[str]


@app.post("/predict/batch")
def batch_predict(request: BatchPredictRequest):
    if not request.texts:
        raise HTTPException(
            status_code=400,
            detail="文本列表不能为空"
        )

    if any(not text.strip() for text in request.texts):
        raise HTTPException(
            status_code=400,
            detail="每条文本都不能为空"
        )

    predictions = predict_batch(request.texts)

    results = []

    for prediction in predictions:
        prediction["inference_mode"] = "INT8" if USE_INT8 else "FP32"
        prediction["checkpoint_sha256"] = CHECKPOINT_SHA256
        results.append(
            add_decision(prediction, DECISION_THRESHOLD)
        )

    return {"count": len(results), "results": results}