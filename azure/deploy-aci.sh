#!/bin/bash

# Azure Container Instances Deployment Script
# This script deploys the Heart Disease Prediction API to Azure Container Instances

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration variables
RESOURCE_GROUP="mlops-heart-disease-rg"
LOCATION="eastus"
CONTAINER_NAME="heart-disease-api"
IMAGE_NAME="heart-disease-api:latest"
DNS_NAME_LABEL="heart-disease-api-${RANDOM}"
PORT=8000
CPU=1
MEMORY=1.5

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Azure Container Instances Deployment${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}Error: Azure CLI is not installed.${NC}"
    echo "Please install it from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if logged in to Azure
echo -e "\n${YELLOW}Checking Azure login status...${NC}"
if ! az account show &> /dev/null; then
    echo -e "${YELLOW}Not logged in to Azure. Logging in...${NC}"
    az login
else
    echo -e "${GREEN}Already logged in to Azure${NC}"
    az account show --output table
fi

# Create resource group
echo -e "\n${YELLOW}Creating resource group: ${RESOURCE_GROUP}...${NC}"
az group create \
    --name ${RESOURCE_GROUP} \
    --location ${LOCATION} \
    --output table

# Build and push Docker image to Azure Container Registry (optional)
# Uncomment if using ACR instead of local image
# ACR_NAME="mlopsheartdiseaseacr${RANDOM}"
# echo -e "\n${YELLOW}Creating Azure Container Registry...${NC}"
# az acr create \
#     --resource-group ${RESOURCE_GROUP} \
#     --name ${ACR_NAME} \
#     --sku Basic \
#     --admin-enabled true \
#     --output table
#
# echo -e "\n${YELLOW}Building and pushing image to ACR...${NC}"
# az acr build \
#     --registry ${ACR_NAME} \
#     --image ${IMAGE_NAME} \
#     --file Dockerfile \
#     .
#
# ACR_LOGIN_SERVER=$(az acr show --name ${ACR_NAME} --resource-group ${RESOURCE_GROUP} --query "loginServer" --output tsv)
# ACR_USERNAME=$(az acr credential show --name ${ACR_NAME} --query "username" --output tsv)
# ACR_PASSWORD=$(az acr credential show --name ${ACR_NAME} --query "passwords[0].value" --output tsv)
# IMAGE_PATH="${ACR_LOGIN_SERVER}/${IMAGE_NAME}"

# For Docker Hub (alternative)
DOCKER_USERNAME="${DOCKER_USERNAME:-your-dockerhub-username}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-your-dockerhub-password}"
IMAGE_PATH="${DOCKER_USERNAME}/heart-disease-api:latest"

echo -e "\n${YELLOW}Building Docker image locally...${NC}"
docker build -t ${IMAGE_NAME} .

echo -e "\n${YELLOW}Tagging and pushing to Docker Hub...${NC}"
docker tag ${IMAGE_NAME} ${IMAGE_PATH}
docker login -u ${DOCKER_USERNAME} -p ${DOCKER_PASSWORD}
docker push ${IMAGE_PATH}

# Deploy container to ACI
echo -e "\n${YELLOW}Deploying container to Azure Container Instances...${NC}"
az container create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${CONTAINER_NAME} \
    --image ${IMAGE_PATH} \
    --cpu ${CPU} \
    --memory ${MEMORY} \
    --dns-name-label ${DNS_NAME_LABEL} \
    --ports ${PORT} \
    --environment-variables \
        MODEL_PATH="/app/models/rf_heart.joblib" \
        PYTHONUNBUFFERED="1" \
    --restart-policy OnFailure \
    --output table

# Wait for container to be ready
echo -e "\n${YELLOW}Waiting for container to be ready...${NC}"
sleep 10

# Get container details
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Successful!${NC}"
echo -e "${GREEN}========================================${NC}"

FQDN=$(az container show \
    --resource-group ${RESOURCE_GROUP} \
    --name ${CONTAINER_NAME} \
    --query "ipAddress.fqdn" \
    --output tsv)

IP=$(az container show \
    --resource-group ${RESOURCE_GROUP} \
    --name ${CONTAINER_NAME} \
    --query "ipAddress.ip" \
    --output tsv)

echo -e "\n${GREEN}Container Name:${NC} ${CONTAINER_NAME}"
echo -e "${GREEN}Resource Group:${NC} ${RESOURCE_GROUP}"
echo -e "${GREEN}FQDN:${NC} ${FQDN}"
echo -e "${GREEN}IP Address:${NC} ${IP}"
echo -e "${GREEN}Port:${NC} ${PORT}"

echo -e "\n${GREEN}API Endpoints:${NC}"
echo -e "  Health Check: ${GREEN}http://${FQDN}:${PORT}/health${NC}"
echo -e "  Prediction:   ${GREEN}http://${FQDN}:${PORT}/predict${NC}"
echo -e "  Metrics:      ${GREEN}http://${FQDN}:${PORT}/metrics${NC}"

# Test the deployment
echo -e "\n${YELLOW}Testing health endpoint...${NC}"
sleep 5
if curl -f "http://${FQDN}:${PORT}/health" 2>/dev/null; then
    echo -e "${GREEN}✓ Health check successful!${NC}"
else
    echo -e "${RED}✗ Health check failed. Check container logs.${NC}"
fi

echo -e "\n${YELLOW}To view container logs:${NC}"
echo "az container logs --resource-group ${RESOURCE_GROUP} --name ${CONTAINER_NAME}"

echo -e "\n${YELLOW}To test prediction:${NC}"
cat <<EOF
curl -X POST http://${FQDN}:${PORT}/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "data": [{
      "age": 55, "sex": 1, "cp": 3, "trestbps": 140, "chol": 230,
      "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
      "oldpeak": 1.0, "slope": 0, "ca": 0, "thal": 3
    }]
  }'
EOF

echo -e "\n${YELLOW}To delete deployment:${NC}"
echo "az group delete --name ${RESOURCE_GROUP} --yes --no-wait"

echo -e "\n${GREEN}Deployment complete!${NC}"

