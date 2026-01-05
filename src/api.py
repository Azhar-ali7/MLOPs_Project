"""FastAPI app exposing prediction endpoint and health checks with full-stack logging."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd
import os
import logging
import json
import time
import sys
from datetime import datetime
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from .model import ModelWrapper

# ================== LOGGING CONFIGURATION ==================


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging - compatible with ELK stack."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "service": "heart-disease-api",
            "version": "1.0.0",
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        # Add custom dimensions for backward compatibility
        if hasattr(record, "custom_dimensions"):
            log_data["custom_dimensions"] = record.custom_dimensions

        return json.dumps(log_data)


def setup_logging():
    """Configure logging with JSON format for ELK stack and console output."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Create logger
    logger = logging.getLogger("heart-disease-api")
    logger.setLevel(getattr(logging, log_level))
    logger.handlers = []  # Clear existing handlers

    # Console handler with JSON formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

    # File handler for Fluentd tail input
    # Use /app/logs in Docker, ./logs locally
    log_dir = Path("/app/logs") if Path("/app").exists() else Path("./logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError):
        # Fallback to current directory if can't create log dir
        log_dir = Path(".")
    file_handler = logging.FileHandler(log_dir / "api.log")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()

# ================== PROMETHEUS METRICS ==================

# Request metrics
REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["method", "endpoint", "status"])

REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# Prediction metrics
PREDICTION_COUNT = Counter("predictions_total", "Total predictions made", ["result"])

PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds",
    "Prediction latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

PREDICTION_PROBABILITY = Histogram(
    "prediction_probability",
    "Distribution of prediction probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Model metrics
MODEL_LOADED = Gauge("model_loaded", "Whether the model is loaded (1) or not (0)")

ACTIVE_REQUESTS = Gauge("active_requests", "Number of requests currently being processed")

# Batch metrics
BATCH_SIZE = Histogram(
    "prediction_batch_size", "Size of prediction batches", buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
)

# ================== PYDANTIC MODELS ==================


class FeatureInput(BaseModel):
    """Single patient feature input."""

    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0=female, 1=male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=0, description="Resting blood pressure")
    chol: float = Field(..., ge=0, description="Serum cholesterol")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=0, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, description="ST depression")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia")


class PredictRequest(BaseModel):
    """Prediction request with list of feature inputs."""

    data: List[Dict[str, Any]] = Field(..., description="List of patient feature dictionaries")


class PredictResponse(BaseModel):
    """Prediction response."""

    predictions: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    model_loaded: bool
    version: str
    uptime_seconds: float


# ================== APP INITIALIZATION ==================

start_time = time.time()


def get_model_path() -> Path:
    """Get model path from environment or use default."""
    env = os.getenv("MODEL_PATH")
    if env:
        return Path(env)
    return Path("models/random_forest.joblib")


# Initialize model
model = ModelWrapper(get_model_path())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info(
        "Starting Heart Disease Prediction API",
        extra={"extra_data": {"event": "startup", "model_path": str(model.model_path)}},
    )

    try:
        model.load()
        MODEL_LOADED.set(1)
        logger.info(
            "Model loaded successfully",
            extra={"extra_data": {"event": "model_loaded", "model_path": str(model.model_path)}},
        )
    except Exception as e:
        MODEL_LOADED.set(0)
        logger.error(
            f"Failed to load model: {e}", extra={"extra_data": {"event": "model_load_failed", "error": str(e)}}
        )

    yield

    # Shutdown
    logger.info("Shutting down Heart Disease Prediction API", extra={"extra_data": {"event": "shutdown"}})


app = FastAPI(
    title="Heart Disease Predictor",
    version="1.0.0",
    description="ML API for predicting heart disease risk",
    lifespan=lifespan,
)


# ================== MIDDLEWARE ==================


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with structured logging."""
    request_id = f"{time.time_ns()}"
    start_time_req = time.time()

    ACTIVE_REQUESTS.inc()

    # Extract request info
    client_host = request.client.host if request.client else "unknown"

    logger.info(
        "Request received",
        extra={
            "extra_data": {
                "event": "request_start",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": client_host,
                "user_agent": request.headers.get("user-agent", "unknown"),
            }
        },
    )

    try:
        response = await call_next(request)
        duration = time.time() - start_time_req

        # Record metrics
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=response.status_code).inc()

        REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(duration)

        logger.info(
            "Request completed",
            extra={
                "extra_data": {
                    "event": "request_complete",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_seconds": round(duration, 4),
                    "client_ip": client_host,
                }
            },
        )

        return response

    except Exception as e:
        duration = time.time() - start_time_req

        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=500).inc()

        logger.error(
            "Request failed",
            extra={
                "extra_data": {
                    "event": "request_error",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "duration_seconds": round(duration, 4),
                }
            },
            exc_info=True,
        )

        raise
    finally:
        ACTIVE_REQUESTS.dec()


# ================== ENDPOINTS ==================


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Health check endpoint for liveness and readiness probes."""
    uptime = time.time() - start_time

    logger.info(
        "Health check requested",
        extra={
            "extra_data": {
                "event": "health_check",
                "model_loaded": model._model is not None,
                "uptime_seconds": round(uptime, 2),
            }
        },
    )

    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat() + "Z",
        model_loaded=model._model is not None,
        version="1.0.0",
        uptime_seconds=round(uptime, 2),
    )


@app.get("/ready", tags=["Health"])
def ready():
    """Readiness probe - checks if model is loaded and ready to serve."""
    if model._model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {"status": "ready", "timestamp": datetime.utcnow().isoformat() + "Z"}


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics endpoint."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest, request: Request):
    """
    Make heart disease predictions.

    Accepts a list of patient feature dictionaries and returns predictions
    with confidence scores.
    """
    prediction_start = time.time()
    client_host = request.client.host if request.client else "unknown"

    # Record batch size
    batch_size = len(req.data)
    BATCH_SIZE.observe(batch_size)

    logger.info(
        "Prediction request received",
        extra={"extra_data": {"event": "prediction_start", "num_samples": batch_size, "client_ip": client_host}},
    )

    try:
        df = pd.DataFrame(req.data)

        logger.info(
            "DataFrame created",
            extra={"extra_data": {"event": "dataframe_created", "shape": list(df.shape), "columns": list(df.columns)}},
        )

        # Make predictions
        preds = model.predict(df)

        # Record prediction metrics
        duration = time.time() - prediction_start
        PREDICTION_LATENCY.observe(duration)

        for pred in preds:
            result = "positive" if pred["prediction"] == 1 else "negative"
            PREDICTION_COUNT.labels(result=result).inc()
            PREDICTION_PROBABILITY.observe(pred["probability"])

        # Log predictions
        logger.info(
            "Predictions completed",
            extra={
                "extra_data": {
                    "event": "prediction_complete",
                    "num_samples": batch_size,
                    "num_positive": sum(1 for p in preds if p["prediction"] == 1),
                    "num_negative": sum(1 for p in preds if p["prediction"] == 0),
                    "avg_probability": round(sum(p["probability"] for p in preds) / len(preds), 4),
                    "duration_seconds": round(duration, 4),
                    "model_path": str(model.model_path),
                }
            },
        )

        return PredictResponse(
            predictions=preds,
            metadata={
                "num_samples": len(preds),
                "duration_seconds": round(duration, 4),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model_version": "1.0.0",
            },
        )

    except ValueError as e:
        logger.error(
            f"Validation error: {str(e)}",
            extra={"extra_data": {"event": "validation_error", "error": str(e), "num_samples": batch_size}},
        )
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    except Exception as e:
        logger.error(
            f"Prediction error: {str(e)}",
            extra={"extra_data": {"event": "prediction_error", "error": str(e), "num_samples": batch_size}},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/model/info", tags=["Model"])
def model_info():
    """Get information about the loaded model."""
    if model._model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    m = model._model
    info = {
        "model_type": type(m).__name__,
        "model_path": str(model.model_path),
        "features": list(getattr(m, "feature_names_in_", [])),
        "n_features": len(getattr(m, "feature_names_in_", [])),
        "loaded_at": datetime.utcnow().isoformat() + "Z",
    }

    # Add model-specific info
    if hasattr(m, "n_estimators"):
        info["n_estimators"] = m.n_estimators
    if hasattr(m, "max_depth"):
        info["max_depth"] = m.max_depth

    logger.info("Model info requested", extra={"extra_data": {"event": "model_info", "model_type": info["model_type"]}})

    return info


@app.get("/", tags=["Info"])
def root():
    """API root - returns basic information."""
    return {
        "name": "Heart Disease Prediction API",
        "version": "1.0.0",
        "description": "ML API for predicting heart disease risk",
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "metrics": "/metrics",
            "predict": "/predict",
            "model_info": "/model/info",
            "docs": "/docs",
        },
    }
