#!/bin/bash

# ============================================================
# Interactive MLOps Workflow Script
# Guides through complete development workflow with user prompts
# ============================================================

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

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

ask_continue() {
    local prompt="${1:-Continue to next step?}"
    echo ""
    echo -e "${YELLOW}${prompt}${NC}"
    echo -e "Press ${GREEN}Enter${NC} to continue, ${RED}Ctrl+C${NC} to exit, or type ${YELLOW}'skip'${NC} to skip this step"
    read -r response
    if [[ "$response" == "skip" ]]; then
        return 1
    fi
    return 0
}

# ============================================================
# ENVIRONMENT MANAGEMENT
# ============================================================

deactivate_existing_env() {
    if [[ -n "$VIRTUAL_ENV" ]]; then
        print_warning "Virtual environment detected: $VIRTUAL_ENV"
        echo "Deactivating current environment..."
        deactivate 2>/dev/null || true
        unset VIRTUAL_ENV
        print_success "Environment deactivated"
    fi
}

activate_venv() {
    if [[ -f ".venv/bin/activate" ]]; then
        print_step "Activating virtual environment..."
        source .venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_error "Virtual environment not found at .venv/"
        print_warning "Run './scripts/setup.sh' first to create environment"
        exit 1
    fi
}

# ============================================================
# WORKFLOW STEPS
# ============================================================

step_environment_setup() {
    print_header "STEP 1: Environment Setup"
    
    # Deactivate any existing environment
    deactivate_existing_env
    
    # Check if .venv exists
    if [[ ! -d ".venv" ]]; then
        print_step "Creating virtual environment..."
        python3 -m venv .venv
        print_success "Virtual environment created"
    else
        print_success "Virtual environment already exists"
    fi
    
    # Activate environment
    activate_venv
    
    # Upgrade pip
    print_step "Upgrading pip..."
    pip install --upgrade pip -q
    print_success "pip upgraded"
    
    # Install dependencies
    print_step "Installing dependencies..."
    pip install -r requirements.txt -q
    print_success "Dependencies installed"
    
    # Create necessary directories
    print_step "Creating directories..."
    mkdir -p data/raw data/processed models mlruns logs
    print_success "Directories created"
}

step_data_preparation() {
    print_header "STEP 2: Data Preparation"
    
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
    print_header "STEP 3: Model Training"
    
    print_step "Training models with MLflow tracking..."
    python -m src.train --data data/processed/heart_processed.csv --model-dir models --cv 5
    print_success "Models trained successfully"
    
    echo ""
    print_success "Trained models:"
    ls -lh models/*.joblib 2>/dev/null || echo "No models found"
}

step_run_tests() {
    print_header "STEP 4: Run Tests"
    
    print_step "Running pytest suite..."
    pytest tests/ -v --tb=short --cov=src --cov-report=term
    print_success "Tests completed"
}

step_start_api() {
    print_header "STEP 5: Start API Server"
    
    print_warning "API server will start in foreground mode"
    echo "The server will run until you press Ctrl+C"
    echo ""
    echo -e "Once started, you can:"
    echo -e "  - View API docs: ${BLUE}http://localhost:8000/docs${NC}"
    echo -e "  - Test health: ${BLUE}curl http://localhost:8000/health${NC}"
    echo -e "  - Test prediction: ${BLUE}./scripts/quick-test.sh${NC} (in another terminal)"
    echo ""
    
    if ask_continue "Start the API server?"; then
        print_step "Starting uvicorn server..."
        echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
        echo ""
        uvicorn src.api:app --host 0.0.0.0 --port 8000
    fi
}

step_build_docker() {
    print_header "STEP 5: Build Docker Image"
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running"
        print_warning "Please start Docker Desktop and try again"
        return 1
    fi
    
    print_step "Building Docker image..."
    docker build -t heart-disease-api:latest .
    print_success "Docker image built successfully"
    
    echo ""
    print_success "Docker image info:"
    docker images heart-disease-api:latest
}

step_test_docker() {
    print_header "STEP 6: Test Docker Deployment"
    
    print_warning "This will start Docker Compose services"
    
    if ask_continue "Start Docker Compose test?"; then
        ./scripts/test-docker.sh
    else
        print_warning "Skipping Docker test"
    fi
}

step_kubernetes() {
    print_header "STEP 7: Kubernetes Deployment (Docker Desktop)"
    
    echo -e "${YELLOW}This step requires Kubernetes enabled in Docker Desktop${NC}"
    echo ""
    
    # Check for kubectl
    if ! command -v kubectl &> /dev/null; then
        print_warning "kubectl is not installed"
        echo ""
        echo "Kubernetes comes with Docker Desktop."
        echo "Enable it in: Docker Desktop → Settings → Kubernetes → Enable Kubernetes"
        echo ""
        print_warning "Skipping Kubernetes deployment"
        return 1
    fi
    
    print_success "kubectl is installed"
    
    # Check if Kubernetes is running
    if kubectl cluster-info &> /dev/null; then
        CLUSTER_TYPE=$(kubectl config current-context 2>/dev/null || echo "unknown")
        print_success "Kubernetes cluster is running: $CLUSTER_TYPE"
        echo ""
        
        echo "Deployment options:"
        echo "  1. Deploy to Kubernetes"
        echo "  2. View deployment status"
        echo "  3. Delete deployment"
        echo "  4. Skip"
        echo ""
        echo -n "Enter choice (1/2/3/4): "
        read -r choice
        
        case "$choice" in
            1)
                print_step "Building Docker image..."
                docker build -t heart-disease-api:latest .
                print_success "Image built"
                
                echo ""
                print_step "Applying Kubernetes manifests..."
                kubectl apply -f k8s/deployment.yaml
                kubectl apply -f k8s/service.yaml
                kubectl apply -f k8s/ingress.yaml
                
                echo ""
                print_success "Deployed to Kubernetes"
                
                echo ""
                print_step "Waiting for pods to be ready..."
                kubectl wait --for=condition=ready pod -l app=heart-disease-api --timeout=60s 2>/dev/null || true
                
                echo ""
                echo -e "${GREEN}Deployment complete!${NC}"
                echo ""
                echo "Access the API:"
                echo -e "  ${CYAN}kubectl port-forward svc/heart-disease-api 8000:8000${NC}"
                echo ""
                echo "View pods:"
                echo -e "  ${CYAN}kubectl get pods${NC}"
                echo ""
                echo "View services:"
                echo -e "  ${CYAN}kubectl get services${NC}"
                ;;
            2)
                echo ""
                echo "Pods:"
                kubectl get pods -l app=heart-disease-api 2>/dev/null || echo "No pods found"
                echo ""
                echo "Services:"
                kubectl get services heart-disease-api 2>/dev/null || echo "No service found"
                echo ""
                echo "Ingress:"
                kubectl get ingress heart-disease-api 2>/dev/null || echo "No ingress found"
                ;;
            3)
                print_step "Deleting Kubernetes resources..."
                kubectl delete -f k8s/ 2>/dev/null || true
                print_success "Resources deleted"
                ;;
            4)
                print_warning "Skipping Kubernetes deployment"
                return
                ;;
            *)
                print_error "Invalid choice, skipping"
                return
                ;;
        esac
    else
        print_warning "Kubernetes is not running in Docker Desktop"
        echo ""
        echo "To enable Kubernetes:"
        echo "  1. Open Docker Desktop"
        echo "  2. Click Settings (gear icon)"
        echo "  3. Click Kubernetes (left sidebar)"
        echo "  4. Check 'Enable Kubernetes'"
        echo "  5. Click 'Apply & Restart'"
        echo "  6. Wait 2-3 minutes for Kubernetes to start"
        echo ""
        print_warning "Skipping Kubernetes deployment"
        return 1
    fi
}

# ============================================================
# MAIN WORKFLOW
# ============================================================

run_all_steps() {
    # Step 1: Environment Setup
    if ask_continue "Step 1: Setup environment?"; then
        step_environment_setup
    fi
    
    # Step 2: Data Preparation
    if ask_continue "Step 2: Prepare data?"; then
        step_data_preparation
    fi
    
    # Step 3: Model Training
    if ask_continue "Step 3: Train models?"; then
        step_model_training
    fi
    
    # Step 4: Run Tests
    if ask_continue "Step 4: Run tests?"; then
        step_run_tests
    fi
    
    # Step 5: Build Docker
    if ask_continue "Step 5: Build Docker image?"; then
        step_build_docker
    fi
    
    # Step 6: Test Docker
    if ask_continue "Step 6: Test Docker deployment?"; then
        step_test_docker
    fi
    
    # Step 7: Kubernetes (optional)
    if ask_continue "Step 7: Kubernetes deployment? (optional)"; then
        step_kubernetes
    fi
}

run_specific_step() {
    local step=$1
    
    case $step in
        1)
            step_environment_setup
            ;;
        2)
            step_data_preparation
            ;;
        3)
            step_model_training
            ;;
        4)
            step_run_tests
            ;;
        5)
            step_build_docker
            ;;
        6)
            step_test_docker
            ;;
        7)
            step_kubernetes
            ;;
        *)
            print_error "Invalid step number: $step"
            return 1
            ;;
    esac
}

show_step_menu() {
    while true; do
        echo ""
        echo -e "${BLUE}============================================================${NC}"
        echo -e "${BLUE}  Select a Step to Run${NC}"
        echo -e "${BLUE}============================================================${NC}"
        echo ""
        echo "  1. Environment Setup"
        echo "  2. Data Preparation"
        echo "  3. Model Training"
        echo "  4. Run Tests"
        echo "  5. Build Docker Image"
        echo "  6. Test Docker Deployment"
        echo "  7. Kubernetes Deployment"
        echo ""
        echo "  8. Run all remaining steps"
        echo "  9. Exit"
        echo ""
        echo -n "Enter choice (1-9): "
        read -r choice
        
        if [[ "$choice" == "9" ]]; then
            print_warning "Exiting workflow"
            return 0
        elif [[ "$choice" == "8" ]]; then
            return 1  # Signal to run all remaining steps
        elif [[ "$choice" =~ ^[1-7]$ ]]; then
            run_specific_step "$choice"
        else
            print_error "Invalid choice. Please enter 1-9."
        fi
    done
}

main() {
    print_header "MLOps Interactive Workflow"
    
    echo -e "${MAGENTA}This script will guide you through the complete MLOps workflow:${NC}"
    echo ""
    echo "  1. Environment Setup (venv, dependencies)"
    echo "  2. Data Preparation (download, process)"
    echo "  3. Model Training (with MLflow)"
    echo "  4. Run Tests (pytest)"
    echo "  5. Build Docker Image"
    echo "  6. Test Docker Deployment"
    echo "  7. Kubernetes Deployment (optional)"
    echo ""
    echo -e "${YELLOW}Workflow modes:${NC}"
    echo "  a) Run all steps sequentially (with prompts)"
    echo "  b) Choose specific steps to run"
    echo "  c) Start from a specific step"
    echo ""
    echo -n "Select mode (a/b/c): "
    read -r mode
    
    case "$mode" in
        a)
            echo ""
            echo -e "${GREEN}Running all steps sequentially${NC}"
            echo -e "${YELLOW}You can skip any step by typing 'skip' when prompted${NC}"
            echo ""
            if ask_continue "Start the workflow?"; then
                run_all_steps
            else
                print_warning "Workflow cancelled"
                exit 0
            fi
            ;;
        b)
            echo ""
            echo -e "${GREEN}Interactive step selection mode${NC}"
            if show_step_menu; then
                # User chose to exit
                exit 0
            else
                # User chose "Run all remaining steps"
                echo ""
                echo -e "${GREEN}Running all steps sequentially${NC}"
                run_all_steps
            fi
            ;;
        c)
            echo ""
            echo "Start from which step? (1-7): "
            read -r start_step
            
            if [[ ! "$start_step" =~ ^[1-7]$ ]]; then
                print_error "Invalid step number"
                exit 1
            fi
            
            echo ""
            echo -e "${GREEN}Starting from Step $start_step${NC}"
            
            # Run steps from start_step onwards
            for ((i=start_step; i<=7; i++)); do
                if ask_continue "Step $i: Run this step?"; then
                    run_specific_step "$i"
                fi
            done
            ;;
        *)
            print_error "Invalid mode. Please run again and choose a, b, or c."
            exit 1
            ;;
    esac
    
    # Summary
    print_header "Workflow Complete!"
    
    echo -e "${GREEN}✓ All selected steps completed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Access your services:${NC}"
    echo ""
    echo "API (Docker Compose):"
    echo -e "  ${BLUE}http://localhost:8000/docs${NC} - API Documentation"
    echo -e "  ${BLUE}http://localhost:8000/health${NC} - Health Check"
    echo ""
    echo "MLflow UI (Docker Compose):"
    echo -e "  ${BLUE}http://localhost:5050${NC} - Experiment Tracking"
    echo ""
    echo "Prometheus (Docker Compose):"
    echo -e "  ${BLUE}http://localhost:9090${NC} - Metrics"
    echo ""
    echo "Grafana (Docker Compose):"
    echo -e "  ${BLUE}http://localhost:3000${NC} - Dashboards (admin/admin)"
    echo ""
    echo "Streamlit UI (Docker Compose):"
    echo -e "  ${BLUE}http://localhost:8501${NC} - Web Interface"
    echo ""
    echo -e "${YELLOW}Useful commands:${NC}"
    echo ""
    echo "Test API:"
    echo -e "  ${CYAN}./scripts/quick-test.sh${NC}"
    echo ""
    echo "View Docker logs:"
    echo -e "  ${CYAN}docker compose logs -f api${NC}"
    echo ""
    echo "Stop all services:"
    echo -e "  ${CYAN}docker compose down${NC}"
    echo ""
    echo "Access Kubernetes (if deployed):"
    echo -e "  ${CYAN}kubectl port-forward svc/heart-disease-api 8000:8000${NC}"
    echo ""
}

# Run main workflow
main
