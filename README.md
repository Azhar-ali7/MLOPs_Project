# MLOps Assignment 1 Group - 49
## End-to-End Machine Learning Pipeline with CI/CD and Production Deployment

**Group**: 49  
**Repository**: https://github.com/Azhar-ali7/MLOPs_Project  

---

A production-ready MLOps pipeline for heart disease prediction featuring automated CI/CD, experiment tracking, containerized deployment, and comprehensive monitoring.

**🎯 For Evaluators:** See [EVALUATOR_GUIDE.md](EVALUATOR_GUIDE.md) for a quick 3-step evaluation process that works on Mac, Windows, and Linux!

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Project Structure](#project-structure)
5. [Manual Commands](#manual-commands)
6. [Using the Streamlit UI](#using-the-streamlit-ui)
7. [Monitoring & Observability](#monitoring--observability)
8. [API Usage Examples](#api-usage-examples)
9. [Troubleshooting](#troubleshooting)
10. [Exploratory Data Analysis & Modeling](#exploratory-data-analysis--modeling)
11. [Experiment Tracking Summary](#experiment-tracking-summary)
12. [System Architecture](#system-architecture)
13. [CI/CD Pipeline](#cicd-pipeline)

---

## Features

- **ML Models**: Random Forest (86.9% accuracy) & Logistic Regression with hyperparameter tuning
- **API**: FastAPI with health checks and prediction endpoints
- **Experiment Tracking**: MLflow for model versioning and metrics
- **Monitoring**: Prometheus metrics + Grafana dashboards
- **UI**: Streamlit web interface for training and predictions
- **Containerization**: Docker Compose for easy deployment
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Orchestration**: Kubernetes manifests for production deployment

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

### 2. Start with Docker (Recommended - Works on Mac/Windows/Linux)

```bash
# Build Docker images first
docker compose build

# Start all services
docker compose up -d

# Run the complete workflow inside Docker
docker compose exec api bash scripts/docker-workflow.sh
```

The Docker workflow will:
- Download and prepare data
- Train models with MLflow tracking
- Run the complete test suite
- Verify all services are healthy

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

### Docker Commands (Recommended)

```bash
# Start all services
docker compose up -d

# Run complete workflow inside Docker
docker compose exec api bash scripts/docker-workflow.sh

# Or run individual commands:
docker compose exec api python src/download_data.py --output data/raw/heart.csv
docker compose exec api python src/data --input data/raw/heart.csv --output data/processed/heart_processed.csv
docker compose exec api python src/train --data data/processed/heart_processed.csv --model-dir models
docker compose exec api pytest tests/ -v

# View logs
docker compose logs -f api

# Stop services
docker compose down
```

### Local Development (Optional)

For local Python development without Docker:

```bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

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

### Cross-Platform Notes

**For Windows Users:**
- Use PowerShell or Command Prompt for Docker commands
- All Docker commands work identically on Windows
- No need to run bash scripts directly - use `docker compose exec api bash scripts/docker-workflow.sh`

**For Mac/Linux Users:**
- All commands work as shown
- Can run scripts directly: `./scripts/docker-workflow.sh` (local) or inside Docker (recommended)

### Services Not Starting

```bash
# Check Docker is running
docker ps

# Rebuild and restart
docker compose down -v
docker compose build --no-cache
docker compose up -d
```

### Port Already in Use

```bash
# Find and kill process on port 8000 (Mac/Linux)
lsof -ti:8000 | xargs kill -9

# Windows PowerShell
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process

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
- **Documentation**: See `/docs` folder for detailed guides
- **Notebooks**: See `/notebooks` for EDA and modeling
- **Architecture**: See `/docs/architecture` for system design
- **Full Report**: See `/docs/FINAL_REPORT.md` for comprehensive documentation

---

## Exploratory Data Analysis & Modeling

### Dataset Overview

We utilized the **Heart Disease UCI Dataset** containing 303 patient records with 14 clinical features:
- **Demographics**: Age, Sex
- **Clinical Measurements**: Blood pressure, Cholesterol, Blood sugar
- **Diagnostic Results**: ECG results, Maximum heart rate
- **Exercise Metrics**: Exercise-induced angina, ST depression
- **Target**: Binary classification (0: No disease, 1: Disease present)

### EDA Key Findings

Our exploratory analysis (see `notebooks/01_EDA_Heart_Disease.ipynb`) revealed:

1. **Data Quality**: No missing values, 303 complete records
2. **Class Balance**: 54% positive cases, 46% negative cases (well-balanced)
3. **Age Distribution**: Patients aged 29-77, mean age 54 years, risk increases after 50
4. **Gender Patterns**: 68% male patients, higher disease prevalence in males (55%) vs females (25%)
5. **Key Correlations**: 
   - Strong positive correlation between exercise-induced angina and heart disease
   - Negative correlation between maximum heart rate and disease presence
   - Chest pain type (cp) emerged as a strong predictor

### Model Development & Selection

**Feature Engineering Strategy**:
- **Continuous Features**: Age, blood pressure, cholesterol, max heart rate, ST depression (standardized for Logistic Regression)
- **Categorical Features**: Sex, chest pain type, fasting blood sugar, ECG results, exercise angina (already numeric)
- **Train-Test Split**: 80-20 with stratification to maintain class balance
- **Cross-Validation**: 5-fold stratified cross-validation for robust evaluation

**Models Evaluated**:

1. **Logistic Regression**
   - Linear baseline model for interpretability
   - StandardScaler preprocessing pipeline
   - Hyperparameters tuned: C (regularization), penalty (L1/L2)
   - Performance: 83.6% accuracy, 0.891 ROC-AUC

2. **Random Forest** (Selected for Production)
   - Ensemble method handling non-linear relationships
   - No preprocessing required (tree-based)
   - Hyperparameters tuned: n_estimators, max_depth, min_samples_split
   - Performance: **86.9% accuracy, 0.923 ROC-AUC**
   - Top features: chest pain type, maximum heart rate, ST depression

**Model Selection Rationale**:
- Random Forest selected based on superior performance across all metrics
- Better generalization on test set (86.9% vs 83.6% accuracy)
- Robust to outliers due to ensemble nature
- Provides feature importance for interpretability
- Cross-validation showed consistent performance

**Detailed notebooks**: 
- EDA: `notebooks/01_EDA_Heart_Disease.ipynb`
- Modeling: `notebooks/02_Feature_Engineering_and_Modeling.ipynb`

---

## Experiment Tracking Summary

### MLflow Integration

All model training runs are tracked using MLflow, providing full reproducibility and experiment comparison.

**Logged Information**:
- Model parameters and hyperparameters
- Performance metrics (accuracy, precision, recall, F1, ROC-AUC)
- Model artifacts (serialized models, feature metadata)
- Training metadata (timestamp, duration, dataset version)

**Experiment Organization**:
- Experiment name: `heart-disease`
- Multiple runs comparing different algorithms and hyperparameters
- Model registry for version control

**Key Results Summary**:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 86.9% | 88.1% | 85.4% | 86.7% | 0.923 |
| Logistic Regression | 83.6% | 85.2% | 81.8% | 83.5% | 0.891 |

**Access MLflow UI**: http://localhost:5050 (when running)

**Benefits Achieved**:
- Complete reproducibility of all experiments
- Easy comparison between model iterations
- Automated artifact management
- Version control for production models
- Collaboration-ready experiment tracking

See `docs/FINAL_REPORT.md` Section 5 for detailed experiment tracking documentation.

---

## System Architecture

### High-Level Architecture

Our system follows a modern microservices architecture with the following components:

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
├─────────────────────────────────────────────────────────────┤
│  Streamlit UI (8501)  │  API Documentation (8000/docs)      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
├─────────────────────────────────────────────────────────────┤
│  FastAPI Service (8000) - Prediction & Health Endpoints     │
│  • Load Balanced (3 replicas in K8s)                        │
│  • Health Probes (Liveness & Readiness)                     │
│  • Prometheus Metrics Exposed                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      ML Layer                                │
├─────────────────────────────────────────────────────────────┤
│  MLflow (5050) - Experiment Tracking & Model Registry       │
│  • Model Versioning                                         │
│  • Artifact Storage                                         │
│  • Metrics Logging                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Monitoring Layer                           │
├─────────────────────────────────────────────────────────────┤
│  Prometheus (9090) - Metrics Collection                     │
│  Grafana (3000) - Dashboards & Visualization               │
│  Alertmanager (9093) - Alert Management                    │
└─────────────────────────────────────────────────────────────┘
```

**Component Details**:

1. **API Service**: FastAPI application serving predictions with automatic OpenAPI documentation
2. **Streamlit UI**: User-friendly interface for model training, predictions, and monitoring
3. **MLflow**: Centralized experiment tracking and model registry
4. **Prometheus**: Time-series metrics collection from all services
5. **Grafana**: Real-time monitoring dashboards
6. **Alertmanager**: Alert routing and management

**Deployment Options**:
- **Docker Compose**: Local development and testing (6 interconnected services)
- **Kubernetes**: Production deployment with auto-scaling and high availability

**Detailed Architecture Documentation**: See `docs/architecture/ARCHITECTURE.md`

---

## CI/CD Pipeline

### GitHub Actions Workflow

Our automated CI/CD pipeline runs on every push to `main` and on all pull requests:

**Pipeline Stages**:

1. **Lint** - Code quality checks (Flake8, Black)
2. **Test** - Automated testing with Pytest (87% coverage)
3. **Train** - Model training with MLflow logging
4. **Build** - Docker image creation and registry push

**Workflow File**: `.github/workflows/ci.yml`
