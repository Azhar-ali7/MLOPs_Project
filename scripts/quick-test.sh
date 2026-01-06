#!/bin/bash

# ============================================================
# Quick Test Script - Fast API Testing
# Tests core functionality in under 30 seconds
# ============================================================

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

API_URL="${1:-http://localhost:8000}"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Quick API Test${NC}"
echo -e "${BLUE}  URL: ${API_URL}${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Check if API is running
echo -n "Checking API availability... "
if curl -s -f "${API_URL}/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ API is running${NC}"
else
    echo -e "${RED}✗ API is not responding${NC}"
    echo "Start the API first with: uvicorn src.api:app --host 0.0.0.0 --port 8000"
    exit 1
fi

echo ""
echo -e "${YELLOW}Testing Endpoints:${NC}"
echo "-----------------------------------------------------------"

# Test health endpoint
echo -n "1. Health check... "
HEALTH=$(curl -s "${API_URL}/health")
if echo "$HEALTH" | grep -q "healthy"; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
fi

# Test model info
echo -n "2. Model info... "
INFO=$(curl -s "${API_URL}/model/info")
if echo "$INFO" | grep -q "model_path"; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
fi

# Test prediction endpoint
echo -n "3. Prediction (with confidence)... "
RESPONSE=$(curl -s -X POST "${API_URL}/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "samples": [{
            "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
            "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
            "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
        }]
    }')

if echo "$RESPONSE" | grep -q "prediction" && echo "$RESPONSE" | grep -q "probability"; then
    echo -e "${GREEN}✓ PASS${NC}"
    PRED=$(echo "$RESPONSE" | grep -o '"prediction":[0-9]' | cut -d: -f2)
    PROB=$(echo "$RESPONSE" | grep -o '"probability":[0-9.]*' | cut -d: -f2)
    echo "   Result: prediction=${PRED}, confidence=${PROB}"
else
    echo -e "${RED}✗ FAIL${NC}"
    echo "   Response: $RESPONSE"
fi

# Test metrics endpoint
echo -n "4. Prometheus metrics... "
METRICS=$(curl -s "${API_URL}/metrics")
if echo "$METRICS" | grep -q "api_requests_total"; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
fi

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}Quick test complete!${NC}"
echo ""
echo -e "View API docs: ${BLUE}${API_URL}/docs${NC}"
echo -e "${BLUE}============================================================${NC}"
