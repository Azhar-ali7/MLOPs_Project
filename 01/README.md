# Heart Disease Prediction — MLOps Assignment

This repository contains a reproducible MLOps-ready pipeline for training and serving a classifier that predicts heart disease risk using the UCI Heart Disease dataset.

Structure (key files):
- `src/` — data, training, model, and API code
- `data/` — raw and processed datasets (created at runtime)
- `models/` — trained model artifacts
- `mlruns/` — MLflow tracking store (created at runtime)
- `Dockerfile`, `docker-compose.yml` — containerization
- `.github/workflows/ci.yml` — CI pipeline (runs tests + build)

Quick start (local):
1. Create a virtual environment and install dependencies:
   python -m venv .venv; .\.venv\Scripts\Activate; pip install -r requirements.txt
2. Download data (script supports direct UCI URL or a local CSV):
   python -m src.download_data --output data/raw/heart.csv
3. Prepare and train:
   python -m src.data --input data/raw/heart.csv --output data/processed/heart_processed.csv
   python -m src.train --data data/processed/heart_processed.csv --model-dir models
4. Serve API:
   uvicorn src.api:app --host 0.0.0.0 --port 8000

Docker / Compose:
- Build and run the API with Docker using the provided `Dockerfile`.
- `docker-compose.yml` starts the API and a small MLflow server for tracking experiments.

Notes:
- MLflow tracking is configured to use a local file-based backend by default (`./mlruns`).
- See `src/train.py` to change model and hyperparameters, and `src/api.py` for the prediction contract.
