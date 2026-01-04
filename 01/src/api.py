"""FastAPI app exposing prediction endpoint and health checks."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import os

from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from .model import ModelWrapper


app = FastAPI(title='Heart Disease Predictor')

# Metrics
REQUEST_COUNT = Counter('predict_requests_total', 'Total prediction requests')


class PredictRequest(BaseModel):
    data: list


def get_model_path() -> Path:
    # Prefer explicit env var, otherwise look in models/ folder
    env = os.getenv('MODEL_PATH')
    if env:
        return Path(env)
    # default path
    return Path('models/rf_heart.joblib')


model = ModelWrapper(get_model_path())


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/metrics')
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post('/predict')
def predict(req: PredictRequest):
    REQUEST_COUNT.inc()
    try:
        df = pd.DataFrame(req.data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Invalid input: {e}')

    preds = model.predict(df)
    return {'predictions': preds}
