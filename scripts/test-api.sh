#!/bin/bash

# ============================================================
# Test Script for Local Deployment
# Tests all API endpoints and verifies logging/monitoring
# ============================================================

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

API_URL="${API_URL:-http://localhost:8000}"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Testing Heart Disease Prediction API${NC}"
echo -e "${BLUE}  API URL: ${API_URL}${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Function to check endpoint
check_endpoint() {
    local name=$1
    local method=$2
    local endpoint=$3
    local data=$4
    local expected_status=$5
    
    echo -n "Testing ${name}... "
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "${API_URL}${endpoint}")
    else
        response=$(curl -s -w "\n%{http_code}" -X POST "${API_URL}${endpoint}" \
            -H "Content-Type: application/json" \
            -d "${data}")
    fi
    
    status_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$status_code" = "$expected_status" ]; then
        echo -e "${GREEN}✓ PASS${NC} (HTTP ${status_code})"
        return 0
    else
        echo -e "${RED}✗ FAIL${NC} (Expected ${expected_status}, got ${status_code})"
        echo "Response: ${body}"
        return 1
    fi
}

# Track test results
PASSED=0
FAILED=0

run_test() {
    if "$@"; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
}

echo -e "${YELLOW}1. Health & Status Endpoints${NC}"
echo "-----------------------------------------------------------"
run_test check_endpoint "Root endpoint" "GET" "/" "" "200"
run_test check_endpoint "Health check" "GET" "/health" "" "200"
run_test check_endpoint "Readiness probe" "GET" "/ready" "" "200"
run_test check_endpoint "Prometheus metrics" "GET" "/metrics" "" "200"
run_test check_endpoint "Model info" "GET" "/model/info" "" "200"

echo ""
echo -e "${YELLOW}2. Prediction Endpoints${NC}"
echo "-----------------------------------------------------------"

# Single prediction
SINGLE_PREDICTION='{
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
    "oldpeak": 1.5,
    "slope": 1,
    "ca": 0,
    "thal": 2
  }]
}'
run_test check_endpoint "Single prediction" "POST" "/predict" "$SINGLE_PREDICTION" "200"

# Batch prediction
BATCH_PREDICTION='{
  "data": [
    {"age": 55, "sex": 1, "cp": 3, "trestbps": 140, "chol": 230, "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2},
    {"age": 45, "sex": 0, "cp": 1, "trestbps": 120, "chol": 200, "fbs": 0, "restecg": 1, "thalach": 170, "exang": 0, "oldpeak": 0.5, "slope": 2, "ca": 0, "thal": 2},
    {"age": 65, "sex": 1, "cp": 2, "trestbps": 160, "chol": 280, "fbs": 1, "restecg": 0, "thalach": 130, "exang": 1, "oldpeak": 2.5, "slope": 0, "ca": 2, "thal": 3}
  ]
}'
run_test check_endpoint "Batch prediction (3 samples)" "POST" "/predict" "$BATCH_PREDICTION" "200"

echo ""
echo -e "${YELLOW}3. Error Handling${NC}"
echo "-----------------------------------------------------------"

# Invalid input - missing fields (should return 400 Bad Request)
INVALID_INPUT='{"data": [{"age": 55}]}'
run_test check_endpoint "Invalid input (missing fields)" "POST" "/predict" "$INVALID_INPUT" "400"

# Empty data (should return 400 Bad Request)
EMPTY_DATA='{"data": []}'
run_test check_endpoint "Empty data array" "POST" "/predict" "$EMPTY_DATA" "400"

echo ""
echo -e "${YELLOW}4. OpenAPI Documentation${NC}"
echo "-----------------------------------------------------------"
run_test check_endpoint "OpenAPI schema" "GET" "/openapi.json" "" "200"

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "  Passed: ${GREEN}${PASSED}${NC}"
echo -e "  Failed: ${RED}${FAILED}${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed! ✓${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Check the output above.${NC}"
    exit 1
fi
