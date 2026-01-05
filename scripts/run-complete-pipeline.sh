#!/bin/bash

# ============================================================
# COMPLETE MLOPS PIPELINE EXECUTION SCRIPT
# Heart Disease Prediction - End-to-End Automation
# ============================================================
# This script runs the entire MLOps pipeline from data to deployment
# ============================================================

set -e  # Exit on any error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${CYAN}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# ============================================================
# MAIN PIPELINE
# ============================================================

print_header "MLOps Pipeline - Complete Execution"
echo -e "${MAGENTA}This script will execute the complete MLOps pipeline:${NC}"
echo "  1. Environment Setup"
echo "  2. Data Acquisition"
echo "  3. Data Processing"
echo "  4. Model Training"
echo "  5. Unit Testing"
echo "  6. Code Quality Checks"
echo "  7. Docker Build"
echo "  8. API Testing"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# ============================================================
# STEP 1: ENVIRONMENT SETUP
# ============================================================

print_header "STEP 1: Environment Setup"

print_step "Checking Python version..."
python_version=$(python3 --version 2>&1)
echo "$python_version"

if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 11) else 1)'; then
    print_success "Python version OK"
else
    print_error "Python 3.11+ required"
    exit 1
fi

print_step "Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

print_step "Activating virtual environment..."
source .venv/bin/activate
print_success "Virtual environment activated"

print_step "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
print_success "Dependencies installed"

# ============================================================
# STEP 2: DATA ACQUISITION
# ============================================================

print_header "STEP 2: Data Acquisition"

print_step "Creating data directories..."
mkdir -p data/raw data/processed
print_success "Directories created"

print_step "Downloading Heart Disease dataset..."
if [ -f "data/raw/heart.csv" ]; then
    print_warning "Dataset already exists, skipping download"
else
    python -m src.download_data --output data/raw/heart.csv
    print_success "Dataset downloaded"
fi

print_step "Verifying dataset..."
if [ -f "data/raw/heart.csv" ]; then
    lines=$(wc -l < data/raw/heart.csv)
    echo "  Dataset has $lines lines"
    print_success "Dataset verified"
else
    print_error "Dataset not found"
    exit 1
fi

# ============================================================
# STEP 3: DATA PROCESSING
# ============================================================

print_header "STEP 3: Data Processing"

print_step "Processing and cleaning data..."
python -m src.data --input data/raw/heart.csv --output data/processed/heart_processed.csv
print_success "Data processed"

print_step "Verifying processed data..."
if [ -f "data/processed/heart_processed.csv" ]; then
    processed_lines=$(wc -l < data/processed/heart_processed.csv)
    echo "  Processed data has $processed_lines lines"
    print_success "Processed data verified"
else
    print_error "Processed data not found"
    exit 1
fi

# ============================================================
# STEP 4: MODEL TRAINING
# ============================================================

print_header "STEP 4: Model Training"

print_step "Creating models directory..."
mkdir -p models
print_success "Models directory ready"

print_step "Training models (this may take a few minutes)..."
echo "  - Training Random Forest"
echo "  - Training Logistic Regression"
echo "  - Running cross-validation"
echo "  - Logging to MLflow"

python -m src.train \
    --data data/processed/heart_processed.csv \
    --model-dir models \
    --cv 5

print_success "Models trained and saved"

print_step "Verifying trained models..."
if [ -f "models/random_forest.joblib" ] && [ -f "models/logistic_regression.joblib" ]; then
    print_success "Model files found"
    echo "  - models/random_forest.joblib"
    echo "  - models/logistic_regression.joblib"
else
    print_error "Model files not found"
    exit 1
fi

if [ -f "models/model_summary.json" ]; then
    print_success "Model summary generated"
    echo "Model Performance:"
    cat models/model_summary.json | python -m json.tool | grep -A 5 "test_performance"
else
    print_warning "Model summary not found"
fi

# ============================================================
# STEP 5: UNIT TESTING
# ============================================================

print_header "STEP 5: Unit Testing"

print_step "Running unit tests..."
# Set PYTHONPATH to include project root so tests can import src
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
pytest tests/ -v --tb=short --cov=src --cov-report=term --cov-report=html

if [ $? -eq 0 ]; then
    print_success "All tests passed"
    print_step "Coverage report generated at: htmlcov/index.html"
else
    print_error "Tests failed"
    exit 1
fi

# ============================================================
# STEP 6: CODE QUALITY CHECKS
# ============================================================

print_header "STEP 6: Code Quality Checks"

print_step "Running flake8 linter..."
if flake8 src/ tests/ --count --show-source --statistics --max-line-length=120; then
    print_success "Linting passed"
else
    print_warning "Linting issues found (non-critical)"
fi

print_step "Checking code formatting..."
if black --check --line-length=120 src/ tests/; then
    print_success "Code formatting OK"
else
    print_warning "Code formatting issues (run: black src/ tests/)"
fi

# ============================================================
# STEP 7: DOCKER BUILD
# ============================================================

print_header "STEP 7: Docker Build"

print_step "Checking Docker availability..."
if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Please install Docker Desktop"
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    print_error "Docker daemon not running. Please start Docker Desktop"
    exit 1
fi
print_success "Docker is available"

print_step "Building Docker image..."
docker build -t heart-disease-api:latest . --quiet
print_success "Docker image built: heart-disease-api:latest"

print_step "Verifying Docker image..."
if docker images | grep -q heart-disease-api; then
    image_size=$(docker images heart-disease-api:latest --format "{{.Size}}")
    echo "  Image size: $image_size"
    print_success "Docker image verified"
else
    print_error "Docker image not found"
    exit 1
fi

# ============================================================
# STEP 8: API TESTING
# ============================================================

print_header "STEP 8: API Testing"

print_step "Starting API container..."
docker run -d \
    --name heart-api-test \
    -p 8000:8000 \
    -v "$(pwd)/models:/app/models:ro" \
    heart-disease-api:latest

print_step "Waiting for API to be ready..."
sleep 8

print_step "Testing health endpoint..."
if curl -f -s http://localhost:8000/health > /dev/null; then
    health_response=$(curl -s http://localhost:8000/health)
    echo "  Response: $health_response"
    print_success "Health check passed"
else
    print_error "Health check failed"
    docker logs heart-api-test
    docker stop heart-api-test
    docker rm heart-api-test
    exit 1
fi

print_step "Testing prediction endpoint..."
prediction_response=$(curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "data": [{
            "age": 55, "sex": 1, "cp": 3, "trestbps": 140,
            "chol": 230, "fbs": 0, "restecg": 0, "thalach": 150,
            "exang": 0, "oldpeak": 1.0, "slope": 0, "ca": 0, "thal": 3
        }]
    }')

if echo "$prediction_response" | grep -q "predictions"; then
    echo "  Response: $prediction_response" | python -m json.tool 2>/dev/null || echo "$prediction_response"
    print_success "Prediction test passed"
else
    print_error "Prediction test failed"
    docker logs heart-api-test
    docker stop heart-api-test
    docker rm heart-api-test
    exit 1
fi

print_step "Testing metrics endpoint..."
if curl -f -s http://localhost:8000/metrics > /dev/null; then
    print_success "Metrics endpoint accessible"
else
    print_warning "Metrics endpoint not accessible"
fi

print_step "Cleaning up test container..."
docker stop heart-api-test > /dev/null
docker rm heart-api-test > /dev/null
print_success "Test container removed"

# ============================================================
# PIPELINE SUMMARY
# ============================================================

print_header "Pipeline Execution Complete!"

echo -e "${GREEN}✓ All steps completed successfully!${NC}"
echo ""
echo -e "${CYAN}Summary of Generated Artifacts:${NC}"
echo "  📁 Data:"
echo "     - data/raw/heart.csv (raw dataset)"
echo "     - data/processed/heart_processed.csv (processed dataset)"
echo ""
echo "  🤖 Models:"
echo "     - models/random_forest.joblib"
echo "     - models/logistic_regression.joblib"
echo "     - models/model_summary.json"
echo "     - models/features.json"
echo ""
echo "  🧪 Test Results:"
echo "     - htmlcov/index.html (coverage report)"
echo ""
echo "  🐳 Docker:"
echo "     - heart-disease-api:latest (Docker image)"
echo ""
echo -e "${CYAN}MLflow Experiments:${NC}"
echo "  View experiments: mlflow ui"
echo "  Access at: http://localhost:5000"
echo ""
echo -e "${CYAN}Next Steps:${NC}"
echo "  1. View coverage: open htmlcov/index.html"
echo "  2. View MLflow: mlflow ui"
echo "  3. Deploy full stack: ./scripts/deploy-docker.sh"
echo "  4. Deploy to K8s: ./scripts/deploy-k8s.sh"
echo ""
echo -e "${MAGENTA}Pipeline execution time: ${SECONDS} seconds${NC}"
echo ""
print_success "Ready for deployment!"

