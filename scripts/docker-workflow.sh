#!/bin/bash

# ============================================================
# Docker MLOps Workflow Script
# Simplified workflow for running INSIDE Docker containers
# ============================================================

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

cd /app

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
# WORKFLOW STEPS
# ============================================================

step_data_preparation() {
    print_header "STEP 1: Data Preparation"
    
    # Download data
    print_step "Downloading dataset..."
    if [[ -f "data/raw/heart.csv" ]]; then
        print_warning "Dataset already exists, skipping download"
    else
        python -m src.download_data --output data/raw/heart.csv
        print_success "Dataset downloaded"
    fi
    
    # Process data
    print_step "Processing dataset..."
    python -m src.data --input data/raw/heart.csv --output data/processed/heart_processed.csv
    print_success "Dataset processed"
}

step_model_training() {
    print_header "STEP 2: Model Training"
    
    print_step "Training models with MLflow tracking..."
    python -m src.train --data data/processed/heart_processed.csv --model-dir models --cv 5
    print_success "Models trained successfully"
    
    echo ""
    print_success "Trained models:"
    ls -lh models/*.joblib 2>/dev/null || echo "No models found"
}

step_run_tests() {
    print_header "STEP 3: Run Tests"
    
    print_step "Running pytest suite..."
    pytest tests/ -v --tb=short --cov=src --cov-report=term
    print_success "Tests completed"
}

step_check_services() {
    print_header "STEP 4: Check Services"
    
    print_step "Checking API health..."
    if curl -f http://localhost:8000/health 2>/dev/null; then
        print_success "API is healthy"
    else
        print_warning "API not responding (this is normal if API container is running separately)"
    fi
    
    echo ""
    print_step "MLflow UI: http://localhost:5050"
    print_step "Prometheus: http://localhost:9090"
    print_step "Grafana: http://localhost:3000"
    print_step "Streamlit UI: http://localhost:8501"
}

# ============================================================
# MAIN WORKFLOW
# ============================================================

main() {
    print_header "Docker MLOps Workflow"
    
    echo -e "${CYAN}Running inside Docker container${NC}"
    echo -e "${YELLOW}All dependencies are already installed!${NC}"
    echo ""
    echo "This workflow will:"
    echo "  1. Download and prepare data"
    echo "  2. Train ML models with MLflow"
    echo "  3. Run test suite"
    echo "  4. Check service health"
    echo ""
    
    # Confirm start
    echo -n "Press Enter to start workflow (Ctrl+C to cancel): "
    read -r
    
    # Run all steps
    step_data_preparation
    step_model_training
    step_run_tests
    step_check_services
    
    # Summary
    print_header "Workflow Complete!"
    
    echo -e "${GREEN}✓ All steps completed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo ""
    echo "View MLflow experiments:"
    echo -e "  ${CYAN}Open browser: http://localhost:5050${NC}"
    echo ""
    echo "Test API predictions:"
    echo -e "  ${CYAN}curl -X POST http://localhost:8000/predict \\${NC}"
    echo -e "    ${CYAN}-H 'Content-Type: application/json' \\${NC}"
    echo -e "    ${CYAN}-d '{\"age\": 55, \"sex\": 1, \"cp\": 1, \"trestbps\": 130, ...}'${NC}"
    echo ""
    echo "View Streamlit UI:"
    echo -e "  ${CYAN}Open browser: http://localhost:8501${NC}"
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Run main workflow
main
