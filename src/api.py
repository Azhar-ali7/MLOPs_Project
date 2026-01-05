"""FastAPI app exposing prediction endpoint and health checks."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import os
import logging
import json
import time
from datetime import datetime

from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from .model import ModelWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Azure Application Insights integration (optional)
try:
    from opencensus.ext.azure.log_exporter import AzureLogHandler
    connection_string = os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')
    if connection_string:
        logger.addHandler(AzureLogHandler(connection_string=connection_string))
        logger.info("Azure Application Insights enabled")
except ImportError:
    logger.info("Azure Application Insights not available (opencensus-ext-azure not installed)")


app = FastAPI(title='Heart Disease Predictor', version='1.0.0')

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


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log response
    logger.info(f"Response: {response.status_code} - Duration: {duration:.3f}s")
    
    return response


@app.get('/health')
def health():
    """Health check endpoint."""
    logger.info("Health check requested")
    return {
        'status': 'ok',
        'timestamp': datetime.utcnow().isoformat(),
        'model_loaded': model._model is not None
    }


@app.get('/metrics')
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post('/predict')
def predict(req: PredictRequest, request: Request):
    """Make heart disease predictions."""
    REQUEST_COUNT.inc()
    start_time = time.time()
    
    # Log request details
    logger.info(f"Prediction request received from {request.client.host if request.client else 'unknown'}")
    logger.info(f"Input data: {len(req.data)} samples")
    
    try:
        df = pd.DataFrame(req.data)
        logger.info(f"DataFrame created with shape: {df.shape}")
        
        # Make predictions
        preds = model.predict(df)
        
        # Calculate metrics
        duration = time.time() - start_time
        
        # Log predictions
        logger.info(f"Predictions made successfully in {duration:.3f}s")
        logger.info(f"Predictions: {preds}")
        
        # Log to Azure Application Insights (structured)
        properties = {
            'num_samples': len(req.data),
            'duration_seconds': duration,
            'model_path': str(model.model_path),
            'timestamp': datetime.utcnow().isoformat()
        }
        logger.info(f"Prediction completed", extra={'custom_dimensions': properties})
        
        return {
            'predictions': preds,
            'metadata': {
                'num_samples': len(preds),
                'duration_seconds': round(duration, 3),
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f'Invalid input: {e}')
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f'Prediction failed: {e}')
