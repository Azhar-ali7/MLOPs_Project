#!/bin/bash

# ============================================================
# Docker Deployment Test Script
# Tests Docker Compose deployment
# ============================================================

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Docker Compose Deployment Test${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Function to check if service is healthy
check_service() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=0
    
    echo -n "Waiting for $service to be ready... "
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Ready${NC}"
            return 0
        fi
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}✗ Failed${NC}"
    return 1
}

# Detect docker-compose command
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo -e "${RED}Error: docker-compose is not installed.${NC}"
    exit 1
fi

# Start Docker Compose
echo -e "${YELLOW}Starting Docker Compose services...${NC}"
$COMPOSE_CMD up -d

echo ""
echo -e "${YELLOW}Checking services...${NC}"
echo "-----------------------------------------------------------"

# Wait for services to be ready
check_service "API" "http://localhost:8000/health"
check_service "MLflow" "http://localhost:5050"
check_service "Prometheus" "http://localhost:9090/-/healthy"
check_service "Grafana" "http://localhost:3000/api/health"

echo ""
echo -e "${YELLOW}Testing API endpoint...${NC}"
echo "-----------------------------------------------------------"

# Test prediction
echo "Making prediction request..."
RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "samples": [{
            "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
            "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
            "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
        }]
    }')

if echo "$RESPONSE" | grep -q "prediction" && echo "$RESPONSE" | grep -q "probability"; then
    echo -e "${GREEN}✓ Prediction successful${NC}"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
else
    echo -e "${RED}✗ Prediction failed${NC}"
    echo "$RESPONSE"
fi

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}Docker Compose test complete!${NC}"
echo ""
echo -e "Access services:"
echo -e "  API:        ${BLUE}http://localhost:8000/docs${NC}"
echo -e "  MLflow:     ${BLUE}http://localhost:5050${NC}"
echo -e "  Prometheus: ${BLUE}http://localhost:9090${NC}"
echo -e "  Grafana:    ${BLUE}http://localhost:3000${NC} (admin/admin123)"
echo -e "  Streamlit:  ${BLUE}http://localhost:8501${NC}"
echo ""
echo -e "View logs: ${YELLOW}docker-compose logs -f api${NC}"
echo -e "Stop services: ${YELLOW}docker-compose down${NC}"
echo -e "${BLUE}============================================================${NC}"
