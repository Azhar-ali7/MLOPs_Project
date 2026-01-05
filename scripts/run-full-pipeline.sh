#!/bin/bash

# Full MLOps Pipeline Execution Script
# Runs the complete pipeline from data download to API testing

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Full MLOps Pipeline Execution${NC}"
echo -e "${GREEN}========================================${NC}"

# Step 1: Data Acquisition
echo -e "\n${YELLOW}Step 1: Data Acquisition${NC}"
echo "Downloading raw dataset..."
python -m src.download_data --output data/raw/heart.csv
echo -e "${GREEN}✓ Data downloaded${NC}"

# Step 2: Data Preprocessing
echo -e "\n${YELLOW}Step 2: Data Preprocessing${NC}"
echo "Processing and cleaning data..."
python -m src.data --input data/raw/heart.csv --output data/processed/heart_processed.csv
echo -e "${GREEN}✓ Data processed${NC}"

# Step 3: Model Training
echo -e "\n${YELLOW}Step 3: Model Training${NC}"
echo "Training Random Forest and Logistic Regression..."
python -m src.train --data data/processed/heart_processed.csv --model-dir models --cv 5
echo -e "${GREEN}✓ Models trained${NC}"

# Step 4: Run Tests
echo -e "\n${YELLOW}Step 4: Running Tests${NC}"
pytest tests/ -v --cov=src --cov-report=term
echo -e "${GREEN}✓ Tests passed${NC}"

# Step 5: Code Quality
echo -e "\n${YELLOW}Step 5: Code Quality Checks${NC}"
echo "Running linting..."
flake8 src/ tests/ || echo -e "${YELLOW}Warning: Some linting issues found${NC}"
echo -e "${GREEN}✓ Code quality checked${NC}"

# Step 6: Build Docker Image
echo -e "\n${YELLOW}Step 6: Building Docker Image${NC}"
docker build -t heart-disease-api:latest .
echo -e "${GREEN}✓ Docker image built${NC}"

# Step 7: Test API Locally
echo -e "\n${YELLOW}Step 7: Testing API Locally${NC}"
echo "Starting API in background..."
docker run -d --name test-api -p 8000:8000 heart-disease-api:latest
sleep 5

echo "Testing health endpoint..."
if curl -f http://localhost:8000/health 2>/dev/null; then
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${RED}✗ Health check failed${NC}"
fi

echo "Testing prediction endpoint..."
if curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{
      "age": 55, "sex": 1, "cp": 3, "trestbps": 140, "chol": 230,
      "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
      "oldpeak": 1.0, "slope": 0, "ca": 0, "thal": 3
    }]
  }' 2>/dev/null | grep -q "predictions"; then
    echo -e "${GREEN}✓ Prediction endpoint working${NC}"
else
    echo -e "${RED}✗ Prediction endpoint failed${NC}"
fi

# Cleanup
echo -e "\n${YELLOW}Cleaning up test container...${NC}"
docker stop test-api
docker rm test-api

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Pipeline Execution Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\nNext steps:"
echo -e "1. Deploy to Kubernetes: ${YELLOW}kubectl apply -f k8s/${NC}"
echo -e "2. Deploy to Azure ACI: ${YELLOW}./azure/deploy-aci.sh${NC}"
echo -e "3. View MLflow UI: ${YELLOW}mlflow ui${NC}"
echo -e "4. Push to Docker Hub: ${YELLOW}docker push YOUR_USERNAME/heart-disease-api:latest${NC}"

