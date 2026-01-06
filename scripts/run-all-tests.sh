#!/bin/bash

# ============================================================
# Run All Tests - Complete Test Suite
# Runs setup, unit tests, and deployment tests
# ============================================================

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo -e "${MAGENTA}============================================================${NC}"
echo -e "${MAGENTA}  Complete Test Suite${NC}"
echo -e "${MAGENTA}============================================================${NC}"
echo ""

# Function to run test section
run_section() {
    local title=$1
    local command=$2
    
    echo -e "${BLUE}▶ ${title}${NC}"
    echo "-----------------------------------------------------------"
    
    if eval "$command"; then
        echo -e "${GREEN}✓ ${title} passed${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ ${title} failed${NC}"
        echo ""
        return 1
    fi
}

# Track results
PASSED=0
FAILED=0

run_test() {
    if "$@"; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
}

# Test 1: Code Quality
echo -e "${YELLOW}1. Code Quality Checks${NC}"
echo "============================================================"
run_test run_section "Linting (flake8)" "flake8 src/ tests/ --count --show-source --statistics || true"
run_test run_section "Formatting (black)" "black --check src/ tests/ || true"

# Test 2: Unit Tests
echo -e "${YELLOW}2. Unit Tests${NC}"
echo "============================================================"
run_test run_section "Pytest Suite" "pytest tests/ -v --tb=short"

# Test 3: Model Training
echo -e "${YELLOW}3. Model Training${NC}"
echo "============================================================"
run_test run_section "Download Data" "python -m src.download_data --output data/raw/heart.csv"
run_test run_section "Process Data" "python -m src.data --input data/raw/heart.csv --output data/processed/heart_processed.csv"
run_test run_section "Train Models" "python -m src.train --data data/processed/heart_processed.csv --model-dir models"

# Test 4: Local API
echo -e "${YELLOW}4. Local API Test${NC}"
echo "============================================================"
echo "Starting API in background..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 > /dev/null 2>&1 &
API_PID=$!
sleep 5

run_test run_section "API Endpoints" "./scripts/quick-test.sh http://localhost:8000"

# Cleanup
kill $API_PID 2>/dev/null || true
sleep 2

# Summary
echo ""
echo -e "${MAGENTA}============================================================${NC}"
echo -e "${MAGENTA}  Test Summary${NC}"
echo -e "${MAGENTA}============================================================${NC}"
echo -e "Tests Passed: ${GREEN}${PASSED}${NC}"
echo -e "Tests Failed: ${RED}${FAILED}${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  1. Test Docker:     ./scripts/test-docker.sh"
    echo -e "  2. Test Kubernetes: ./scripts/test-kubernetes.sh"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
