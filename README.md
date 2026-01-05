# 🫀 Heart Disease Prediction - MLOps Pipeline

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete MLOps pipeline for heart disease prediction with Docker deployment, full-stack monitoring (ELK + Prometheus/Grafana), and CI/CD automation.

## 📑 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Development Guide](#-development-guide)
- [Deployment](#-deployment)
- [Monitoring](#-monitoring)
- [API Documentation](#-api-documentation)
- [Testing](#-testing)
- [Contributing](#-contributing)

## 🎯 Overview

This project demonstrates a production-ready MLOps pipeline for predicting heart disease using machine learning. It covers the complete ML lifecycle:

1. **Data Acquisition & EDA** - Download and analyze the UCI Heart Disease dataset
2. **Feature Engineering & Modeling** - Train and compare multiple models with hyperparameter tuning
3. **Experiment Tracking** - Track experiments with MLflow
4. **Model Packaging** - Package models for reproducibility
5. **CI/CD Pipeline** - Automated testing with GitHub Actions
6. **Containerization** - Docker for consistent deployment
7. **Production Deployment** - Docker Compose & Kubernetes (Minikube)
8. **Monitoring & Logging** - ELK stack + Prometheus/Grafana

## 🏗 Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MLOps Pipeline                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐          │
│  │   Data    │───▶│   Train   │───▶│   Model   │───▶│   API     │          │
│  │  Ingestion│    │  Pipeline │    │  Registry │    │  Server   │          │
│  └───────────┘    └───────────┘    └───────────┘    └───────────┘          │
│        │               │                                   │                 │
│        ▼               ▼                                   ▼                 │
│  ┌───────────┐    ┌───────────┐                      ┌───────────┐          │
│  │    EDA    │    │   MLflow  │                      │ Prometheus│          │
│  │ Notebooks │    │  Tracking │                      │  Metrics  │          │
│  └───────────┘    └───────────┘                      └───────────┘          │
│                                                            │                 │
│                                                            ▼                 │
│                                                      ┌───────────┐          │
│                                                      │  Grafana  │          │
│                                                      │ Dashboard │          │
│                                                      └───────────┘          │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │                     ELK Stack (Logging)                        │          │
│  │  ┌─────────┐    ┌───────────┐    ┌─────────┐                  │          │
│  │  │ Fluentd │───▶│Elasticsearch│───▶│ Kibana  │                  │          │
│  │  └─────────┘    └───────────┘    └─────────┘                  │          │
│  └───────────────────────────────────────────────────────────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Docker Compose Services

| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | FastAPI prediction service |
| MLflow | 5000 | Experiment tracking UI |
| Elasticsearch | 9200 | Log storage |
| Kibana | 5601 | Log visualization |
| Fluentd | 24224 | Log collection |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Metrics dashboards |
| Alertmanager | 9093 | Alert management |

## ✨ Features

### Machine Learning
- ✅ Binary classification for heart disease prediction
- ✅ Multiple models (Random Forest, Logistic Regression)
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Cross-validation for robust evaluation
- ✅ Feature importance analysis

### MLOps
- ✅ Experiment tracking with MLflow
- ✅ Model versioning and registry
- ✅ Reproducible environments
- ✅ CI/CD with GitHub Actions
- ✅ Docker containerization

### Monitoring & Observability
- ✅ Structured JSON logging
- ✅ Centralized logging with ELK stack
- ✅ Prometheus metrics collection
- ✅ Grafana dashboards
- ✅ Alerting with Alertmanager

### API
- ✅ FastAPI with async support
- ✅ OpenAPI documentation
- ✅ Health check endpoints
- ✅ Batch prediction support
- ✅ Request validation

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/MLOPs_Project.git
cd MLOPs_Project

# Start the full stack
./scripts/deploy-local.sh start

# Test the API
./scripts/test-api.sh
```

**Access the services:**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MLflow: http://localhost:5000
- Kibana: http://localhost:5601
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin123)

### Option 2: Local Development

```bash
# Create conda environment
conda env create -f conda-env.yml
conda activate heart-disease-mlops

# Download data
python -m src.download_data

# Train models
python -m src.train

# Run API
uvicorn src.api:app --reload --port 8000
```

## 📁 Project Structure

```
MLOPs_Project/
├── src/
│   ├── api.py                       # FastAPI application
│   ├── api_local.py                 # API with full logging
│   ├── data.py                      # Data loading utilities
│   ├── model.py                     # Model training/prediction
│   ├── train.py                     # Training script
│   └── download_data.py             # Data download script
├── notebooks/
│   ├── 01_EDA_Heart_Disease.ipynb   # Exploratory Data Analysis
│   └── 02_Feature_Engineering_and_Modeling.ipynb
├── tests/
│   ├── test_api.py                  # API tests
│   ├── test_data.py                 # Data module tests
│   ├── test_model.py                # Model tests
│   └── test_integration.py          # Integration tests
├── k8s/
│   ├── local/                       # Minikube manifests
│   ├── deployment.yaml              # Kubernetes deployment
│   ├── service.yaml                 # Kubernetes service
│   └── README-K8S.md                # K8s deployment guide
├── monitoring/
│   ├── fluentd/                     # Fluentd configuration
│   ├── prometheus/                  # Prometheus configuration
│   ├── grafana/                     # Grafana dashboards
│   └── alertmanager/                # Alertmanager configuration
├── scripts/
│   ├── deploy-local.sh              # Local deployment script
│   ├── deploy-k8s-local.sh          # Minikube deployment
│   └── test-api.sh                  # API testing script
├── data/
│   ├── raw/                         # Raw data
│   └── processed/                   # Processed data
├── models/                          # Saved models
├── docs/                            # Documentation
├── docker-compose.local.yml         # Full stack deployment
├── Dockerfile.local                 # Docker image for local
├── requirements.txt                 # Python dependencies
├── requirements-local.txt           # Local deployment deps
└── README.md                        # This file
```

## 💻 Development Guide

### Setting Up Development Environment

```bash
# Using conda
conda env create -f conda-env.yml
conda activate heart-disease-mlops

# Or using pip
pip install -r requirements.txt
```

### Running the Notebooks

```bash
# Start Jupyter
jupyter notebook notebooks/
```

### Training Models

```bash
# Train with MLflow tracking
python -m src.train

# View experiments
mlflow ui --port 5000
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🚢 Deployment

### Docker Compose Deployment

```bash
# Start all services
./scripts/deploy-local.sh start

# View logs
./scripts/deploy-local.sh logs

# Stop services
./scripts/deploy-local.sh stop

# Clean up
./scripts/deploy-local.sh clean
```

### Kubernetes Deployment (Minikube)

```bash
# Deploy to Minikube
./scripts/deploy-k8s-local.sh deploy

# Check status
./scripts/deploy-k8s-local.sh status

# Get service URL
minikube service heart-disease-api-service -n heart-disease --url

# Delete deployment
./scripts/deploy-k8s-local.sh delete
```

See [k8s/README-K8S.md](k8s/README-K8S.md) for detailed Kubernetes instructions.

## 📊 Monitoring

### Logging (ELK Stack)

Access Kibana at http://localhost:5601 to:
- Search and filter logs
- Create log visualizations
- Set up log-based alerts

### Metrics (Prometheus + Grafana)

**Prometheus** (http://localhost:9090):
- Query metrics
- View targets and alerts
- Explore metric labels

**Grafana** (http://localhost:3000):
- Pre-configured API dashboard
- Request latency graphs
- Prediction distribution charts
- Login: admin/admin123

### Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `api_requests_total` | Counter | Total requests by endpoint |
| `api_request_latency_seconds` | Histogram | Request latency distribution |
| `api_predictions_total` | Counter | Predictions by result |
| `api_prediction_probability` | Histogram | Prediction probabilities |

## 📚 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Swagger UI |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{
      "age": 55,
      "sex": 1,
      "cp": 2,
      "trestbps": 130,
      "chol": 250,
      "fbs": 0,
      "restecg": 0,
      "thalach": 150,
      "exang": 0,
      "oldpeak": 1.0,
      "slope": 1,
      "ca": 0,
      "thal": 2
    }]
  }'
```

### Response Format

```json
{
  "predictions": [1],
  "probabilities": [[0.25, 0.75]],
  "model": "random_forest"
}
```

## 🧪 Testing

### Unit Tests

```bash
pytest tests/test_data.py -v
pytest tests/test_model.py -v
```

### Integration Tests

```bash
pytest tests/test_api.py -v
pytest tests/test_integration.py -v
```

### API Tests

```bash
# Using test script
./scripts/test-api.sh

# Using curl
curl http://localhost:8000/health
```

## 📖 Documentation

- [Local Deployment Guide](docs/LOCAL_DEPLOYMENT.md)
- [Kubernetes Deployment](k8s/README-K8S.md)
- [Video Demonstration Guide](docs/VIDEO_GUIDE.md)
- [Final Report Template](docs/FINAL_REPORT_TEMPLATE.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Heart Disease dataset
- FastAPI for the excellent web framework
- MLflow for experiment tracking
- The open-source community for amazing tools

---

**Made with ❤️ for MLOps learning**
