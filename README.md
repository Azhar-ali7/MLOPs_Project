# Heart Disease Prediction — Complete MLOps Pipeline

[![CI/CD Pipeline](https://github.com/YOUR_USERNAME/MLOPs_Project/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/YOUR_USERNAME/MLOPs_Project/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready MLOps pipeline for predicting heart disease risk using the UCI Heart Disease dataset. This project demonstrates end-to-end machine learning workflow including data acquisition, exploratory analysis, model training with experiment tracking, automated testing, CI/CD, containerization, Kubernetes orchestration, and cloud deployment on Azure.

## 🎯 Project Overview

**Problem Statement:** Build a machine learning classifier to predict the risk of heart disease based on patient health data, and deploy the solution as a cloud-ready, monitored API.

**Key Features:**
- ✅ Comprehensive EDA with professional visualizations
- ✅ Two ML models (Random Forest & Logistic Regression) with hyperparameter tuning
- ✅ MLflow experiment tracking and model versioning
- ✅ Automated testing with pytest (unit + integration tests)
- ✅ CI/CD pipeline with GitHub Actions
- ✅ Docker containerization for reproducibility
- ✅ Kubernetes deployment manifests (Docker Desktop compatible)
- ✅ Azure Container Instances deployment (budget-friendly cloud)
- ✅ Azure Application Insights monitoring and logging
- ✅ Prometheus metrics endpoint
- ✅ FastAPI for high-performance API serving

## 📊 Architecture

```mermaid
graph TB
    subgraph DataPipeline [Data Pipeline]
        A[UCI Dataset] -->|download| B[Raw Data]
        B -->|process| C[Processed Data]
    end
    
    subgraph Training [Model Training]
        C -->|train| D[GridSearchCV]
        D --> E[Random Forest]
        D --> F[Logistic Regression]
        E --> G[MLflow Tracking]
        F --> G
        G --> H[Best Model]
    end
    
    subgraph CICD [CI/CD Pipeline]
        I[GitHub Push] --> J[GitHub Actions]
        J --> K[Lint & Test]
        K --> L[Build Docker]
        L --> M[Push Registry]
    end
    
    subgraph Deploy [Deployment]
        M --> N[Docker Desktop K8s]
        M --> O[Azure ACI]
        H --> N
        H --> O
    end
    
    subgraph Serving [API Serving]
        N --> P[FastAPI]
        O --> P
        P --> Q[/predict]
        P --> R[/health]
        P --> S[/metrics]
    end
    
    subgraph Monitor [Monitoring]
        S --> T[Prometheus]
        P --> U[Azure Insights]
        U --> V[Dashboard]
        T --> V
    end
```

## 📁 Project Structure

```
MLOPs_Project/
├── .github/
│   └── workflows/
│       └── ci.yml                    # CI/CD pipeline
├── azure/
│   ├── deploy-aci.sh                # Azure deployment script
│   ├── aci-deployment.yaml          # ACI configuration
│   └── README-AZURE.md              # Azure deployment guide
├── k8s/
│   ├── deployment.yaml              # Kubernetes deployment
│   ├── service.yaml                 # Kubernetes service
│   ├── configmap.yaml               # Configuration
│   ├── namespace.yaml               # Namespace definition
│   └── README-K8S.md                # Kubernetes guide
├── notebooks/
│   ├── 01_EDA_Heart_Disease.ipynb   # Exploratory analysis
│   └── 02_Feature_Engineering.ipynb # Model development
├── monitoring/
│   ├── azure-monitor-setup.md       # Monitoring setup guide
│   └── dashboard-config.json        # Dashboard configuration
├── docs/
│   ├── ARCHITECTURE.md              # Architecture documentation
│   ├── AZURE_CLI_COMPLETE_GUIDE.md  # Complete Azure CLI tutorial
│   ├── AZURE_QUICKSTART.md          # Quick Azure deployment guide
│   ├── DEPLOYMENT_INSTRUCTIONS.md   # Deployment instructions
│   ├── FINAL_REPORT_TEMPLATE.md     # Report template
│   └── VIDEO_GUIDE.md               # Video recording guide
├── scripts/
│   ├── setup.sh                     # Automated setup
│   └── run-full-pipeline.sh         # End-to-end pipeline
├── src/
│   ├── api.py                       # FastAPI application
│   ├── data.py                      # Data preprocessing
│   ├── download_data.py             # Data acquisition
│   ├── model.py                     # Model wrapper
│   └── train.py                     # Training pipeline
├── tests/
│   ├── test_api.py                  # API tests
│   ├── test_data.py                 # Data processing tests
│   ├── test_model.py                # Model tests
│   ├── test_train.py                # Training tests
│   └── test_integration.py          # Integration tests
├── .dockerignore                    # Docker ignore rules
├── .flake8                          # Linting configuration
├── pyproject.toml                   # Python project config
├── Dockerfile                       # Docker image definition
├── docker-compose.yml               # Docker Compose configuration
├── requirements.txt                 # Python dependencies
├── conda-env.yml                    # Conda environment
└── README.md                        # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop
- Git
- (Optional) Azure CLI for cloud deployment

### Option 1: Automated Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/MLOPs_Project.git
cd MLOPs_Project

# Run automated setup script
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download and process data
   python -m src.download_data --output data/raw/heart.csv
   python -m src.data --input data/raw/heart.csv --output data/processed/heart_processed.csv

# 4. Train models
python -m src.train --data data/processed/heart_processed.csv --model-dir models --cv 5

# 5. Run tests
pytest tests/ -v --cov=src

# 6. Start API server
   uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## 🔬 Exploratory Data Analysis

Comprehensive EDA notebooks are provided in the `notebooks/` directory:

### 01_EDA_Heart_Disease.ipynb
- Dataset overview and statistics
- Missing value analysis
- Feature distributions (histograms, box plots)
- Correlation heatmap
- Class balance visualization
- Outlier detection

### 02_Feature_Engineering_and_Modeling.ipynb
- Feature engineering strategies
- Model comparison (Random Forest vs Logistic Regression)
- Hyperparameter tuning with GridSearchCV
- Cross-validation results
- ROC curves and confusion matrices
- Feature importance analysis

**To view notebooks:**
```bash
jupyter notebook notebooks/
```

## 🤖 Model Training

The training pipeline supports two models with automated hyperparameter tuning:

**Random Forest:**
- Parameters: n_estimators, max_depth, min_samples_split
- Optimization: ROC-AUC score
- Cross-validation: 5-fold stratified

**Logistic Regression:**
- Pipeline: StandardScaler + LogisticRegression
- Parameters: C (regularization), penalty (L1/L2)
- Solver: liblinear

**MLflow Tracking:**
```bash
# View experiments in MLflow UI
mlflow ui

# Access at: http://localhost:5000
```

All experiments are logged with:
- Hyperparameters
- Cross-validation metrics (accuracy, precision, recall, ROC-AUC)
- Model artifacts
- Feature importance (for Random Forest)

## 🧪 Testing

Comprehensive test suite with pytest:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run integration tests only
pytest tests/test_integration.py -v
```

**Test Coverage:**
- Unit tests for data processing
- Model wrapper tests
- Training pipeline tests
- API endpoint tests
- Integration tests (full pipeline)

## 🐳 Docker Deployment

### Build and Run Locally

```bash
# Build Docker image
docker build -t heart-disease-api:latest .

# Run container
docker run -d -p 8000:8000 --name heart-api heart-disease-api:latest

# Test API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [{"age": 55, "sex": 1, "cp": 3, "trestbps": 140, "chol": 230, "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 0, "ca": 0, "thal": 3}]}'
```

### Docker Compose

```bash
# Start API and MLflow server
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## ☸️ Kubernetes Deployment

Deploy to Docker Desktop Kubernetes or any Kubernetes cluster:

```bash
# Enable Kubernetes in Docker Desktop (Settings → Kubernetes → Enable)

# Deploy all resources
kubectl apply -f k8s/

# Check status
kubectl get all
kubectl get pods
kubectl get services

# Test API
kubectl port-forward service/heart-disease-api-service 8080:80
curl http://localhost:8080/health

# View logs
kubectl logs -l app=heart-disease-api -f

# Scale deployment
kubectl scale deployment heart-disease-api --replicas=3

# Clean up
kubectl delete -f k8s/
```

**See [k8s/README-K8S.md](k8s/README-K8S.md) for detailed instructions.**

## ☁️ Azure Cloud Deployment

Deploy to Azure Container Instances (budget-friendly option):

```bash
# Set credentials
export DOCKER_USERNAME="your-dockerhub-username"
export DOCKER_PASSWORD="your-dockerhub-password"

# Run deployment script
chmod +x azure/deploy-aci.sh
./azure/deploy-aci.sh

# The script will:
# 1. Create Azure Resource Group
# 2. Build and push Docker image
# 3. Deploy to Azure Container Instances
# 4. Set up Application Insights (optional)
# 5. Provide API URL
```

**Manual deployment:**
```bash
# Login to Azure
az login

# Create resource group
az group create --name mlops-heart-disease-rg --location eastus

# Deploy container
az container create \
  --resource-group mlops-heart-disease-rg \
  --name heart-disease-api \
  --image YOUR_USERNAME/heart-disease-api:latest \
  --cpu 1 --memory 1.5 \
  --dns-name-label heart-disease-api-$(date +%s) \
  --ports 8000
```

**See [azure/README-AZURE.md](azure/README-AZURE.md) for complete guide.**

## 📊 Monitoring & Logging

### Azure Application Insights

```bash
# Create Application Insights
az monitor app-insights component create \
  --app heart-disease-api-insights \
  --location eastus \
  --resource-group mlops-heart-disease-rg

# Get connection string
az monitor app-insights component show \
  --app heart-disease-api-insights \
  --resource-group mlops-heart-disease-rg \
  --query "connectionString"
```

### Prometheus Metrics

```bash
# Access metrics endpoint
curl http://localhost:8000/metrics

# Metrics include:
# - predict_requests_total: Total prediction requests
# - Custom application metrics
```

**See [monitoring/azure-monitor-setup.md](monitoring/azure-monitor-setup.md) for setup details.**

## 🔄 CI/CD Pipeline

GitHub Actions workflow automates:

1. **Linting**: Code quality checks with flake8 and black
2. **Testing**: Run pytest with coverage reporting
3. **Training**: Train models on processed data
4. **Building**: Create Docker image
5. **Deployment**: Push to container registry

**Trigger:** Automatic on push to main/master branch

**View:** GitHub → Actions tab

## 📖 API Documentation

### Endpoints

#### GET /health
Health check endpoint

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2026-01-05T10:30:00",
  "model_loaded": true
}
```

#### POST /predict
Make heart disease predictions

**Request:**
```json
{
  "data": [{
    "age": 55,
    "sex": 1,
    "cp": 3,
    "trestbps": 140,
    "chol": 230,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 0,
    "ca": 0,
    "thal": 3
  }]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": 1,
      "probability": 0.85
    }
  ],
  "metadata": {
    "num_samples": 1,
    "duration_seconds": 0.023,
    "timestamp": "2026-01-05T10:30:00"
  }
}
```

#### GET /metrics
Prometheus metrics endpoint

### Interactive API Docs

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📈 Performance Metrics

**Model Performance (Test Set):**
- Random Forest: ROC-AUC = 0.91, Accuracy = 0.87
- Logistic Regression: ROC-AUC = 0.89, Accuracy = 0.85

**API Performance:**
- Average response time: <50ms
- Throughput: >100 requests/second
- Container resources: 1 vCPU, 1.5GB RAM

## 🛠️ Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking (optional)
mypy src/
```

### Adding New Features

1. Create feature branch
2. Implement changes with tests
3. Run tests and linting
4. Create pull request
5. CI/CD runs automatically

## 🔒 Security

- No hardcoded credentials
- Secrets via environment variables
- Docker image scanning (optional)
- Regular dependency updates
- Azure Key Vault for production secrets

## 📚 Additional Documentation

### Core Documentation
- [Architecture Documentation](docs/ARCHITECTURE.md)
- [Video Recording Guide](docs/VIDEO_GUIDE.md)
- [Final Report Template](docs/FINAL_REPORT_TEMPLATE.md)

### Azure Deployment
- **[Azure CLI Complete Guide](docs/AZURE_CLI_COMPLETE_GUIDE.md)** - Comprehensive Azure CLI tutorial
- **[Azure Quick Start Guide](docs/AZURE_QUICKSTART.md)** - 30-minute deployment walkthrough
- [Azure Deployment Guide](azure/README-AZURE.md) - Detailed ACI deployment
- [Monitoring Setup](monitoring/azure-monitor-setup.md) - Application Insights configuration

### Kubernetes Deployment
- [Kubernetes Guide](k8s/README-K8S.md) - Local Kubernetes deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 👥 Authors

- **Your Name** - MLOps Assignment Project

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Heart Disease dataset
- FastAPI for the excellent web framework
- MLflow for experiment tracking
- Azure for cloud infrastructure

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Email: your-email@example.com

---

**Note:** This is an educational project demonstrating MLOps best practices. For production use, additional security hardening, monitoring, and compliance measures should be implemented.
