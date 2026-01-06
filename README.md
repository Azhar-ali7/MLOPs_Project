# Assignment 1 Group 49 - MLOps Pipeline

A complete MLOps pipeline for heart disease prediction with Docker deployment, MLflow experiment tracking, and Prometheus/Grafana monitoring.

## Features

- **ML Models**: Random Forest & Logistic Regression with hyperparameter tuning
- **API**: FastAPI with health checks and prediction endpoints
- **Experiment Tracking**: MLflow for model versioning and metrics
- **Monitoring**: Prometheus metrics + Grafana dashboards
- **UI**: Streamlit web interface for training and predictions
- **Containerization**: Docker Compose for easy deployment

## Prerequisites

Before running the project, ensure you have:

1. **Docker Desktop** installed and running
   - Download from: https://www.docker.com/products/docker-desktop
   - Verify: `docker --version` and `docker compose version`

2. **Kubernetes enabled** in Docker Desktop (optional, for K8s deployment)
   - Open Docker Desktop → Settings → Kubernetes
   - Check "Enable Kubernetes" → Apply & Restart

3. **Python 3.11+** (for local development only)
   - Verify: `python --version`

4. **Git** installed
   - Verify: `git --version`

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd MLOPs_Project
```

### 2. Run the Interactive Setup

```bash
# Make script executable
chmod +x scripts/interactive-workflow.sh

# Activate your Python virtual environment first
source .venv/bin/activate  # or: conda activate your-env

# Run the interactive workflow
./scripts/interactive-workflow.sh
```

The interactive script will guide you through:
- Environment setup (dependencies installation)
- Data preparation
- Model training with MLflow
- Running tests
- Building Docker images
- Starting all services

### 3. Access the Services

Once deployed, access these URLs:

| Service | URL | Credentials |
|---------|-----|-------------|
| **Streamlit UI** | http://localhost:8501 | - |
| **API Docs** | http://localhost:8000/docs | - |
| **MLflow** | http://localhost:5050 | - |
| **Grafana** | http://localhost:3000 | admin/admin123 |
| **Prometheus** | http://localhost:9090 | - |

## Project Structure

```
MLOPs_Project/
├── src/                    # Source code
│   ├── api.py             # FastAPI application
│   ├── data.py            # Data processing
│   ├── model.py           # Model training
│   └── train.py           # Training pipeline
├── ui/                     # Streamlit web interface
│   └── streamlit_app.py
├── tests/                  # Test suite
├── data/                   # Dataset storage
│   ├── raw/               # Raw data
│   └── processed/         # Processed data
├── models/                 # Trained models
├── mlruns/                 # MLflow experiments
├── monitoring/             # Monitoring configs
│   ├── prometheus/
│   └── grafana/
├── k8s/                    # Kubernetes manifests
├── scripts/                # Automation scripts
└── docker-compose.yml      # Service orchestration
```

## Manual Commands

If you prefer manual setup over the interactive script:

### Start All Services

```bash
# Start with Docker Compose
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Train Models Locally

```bash
# Activate environment
source .venv/bin/activate

# Download data
python -m src.download_data --output data/raw/heart.csv

# Process data
python -m src.data --input data/raw/heart.csv --output data/processed/heart_processed.csv

# Train models
python -m src.train --data data/processed/heart_processed.csv --model-dir models
```

### Run Tests

```bash
pytest tests/ -v --cov=src
```

## Using the Streamlit UI

1. Open http://localhost:8501
2. Navigate through pages:
   - **Home**: Quick access to services
   - **Train Models**: Train ML models with custom hyperparameters
   - **MLflow Experiments**: View and compare model runs
   - **Prediction**: Make predictions interactively
   - **Metrics**: View API metrics
   - **Testing**: Run API tests

## Monitoring & Observability

### Prometheus Metrics

Available at http://localhost:9090

Key metrics:
- `api_requests_total` - Total API requests
- `api_request_duration_seconds` - Request latency
- `api_predictions_total` - Prediction counts

### Grafana Dashboards

Login at http://localhost:3000 (admin/admin123)

Pre-configured dashboard shows:
- Request rate and latency
- Prediction distribution
- Error rates
- System metrics

## Kubernetes Deployment (Optional)

If you enabled Kubernetes in Docker Desktop:

```bash
# Build image
docker build -t heart-disease-api:latest .

# Deploy to Kubernetes
kubectl apply -f k8s/

# Check status
kubectl get pods
kubectl get services

# Access API
kubectl port-forward svc/heart-disease-api 8000:8000

# Cleanup
kubectl delete -f k8s/
```

## API Usage Examples

### Health Check

```bash
curl http://localhost:8000/health
```

### Make Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{
      "age": 55, "sex": 1, "cp": 2,
      "trestbps": 130, "chol": 250, "fbs": 0,
      "restecg": 0, "thalach": 150, "exang": 0,
      "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
    }]
  }'
```

## Troubleshooting

### Services Not Starting

```bash
# Check Docker is running
docker ps

# Rebuild and restart
docker compose down -v
docker compose up -d --build
```

### Port Already in Use

```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or change port in docker-compose.yml
```

### Streamlit Loading Issues

```bash
# Restart streamlit service
docker compose restart streamlit

# View logs
docker logs streamlit-ui
```

## Additional Resources

- **Dataset**: UCI Heart Disease Dataset
- **Documentation**: See `/docs` folder
- **Notebooks**: See `/notebooks` for EDA

## License

MIT License

---

Made for MLOps learning
