#!/bin/bash

# ============================================================
# Local Deployment Script for Heart Disease Prediction API
# Full Stack: API + MLflow + ELK Stack + Prometheus + Grafana
# ============================================================

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Heart Disease Prediction API - Local Deployment${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}Error: Docker is not running.${NC}"
        echo "Please start Docker Desktop and try again."
        exit 1
    fi
    echo -e "${GREEN}✓ Docker is running${NC}"
}

# Function to check if docker-compose is available
check_docker_compose() {
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        echo -e "${RED}Error: docker-compose is not installed.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Docker Compose is available${NC}"
}

# Create necessary directories
setup_directories() {
    echo -e "\n${YELLOW}Setting up directories...${NC}"
    mkdir -p logs
    mkdir -p mlruns
    mkdir -p monitoring/grafana/provisioning/datasources
    mkdir -p monitoring/grafana/provisioning/dashboards
    mkdir -p monitoring/grafana/dashboards
    mkdir -p monitoring/prometheus
    mkdir -p monitoring/alertmanager
    mkdir -p monitoring/fluentd
    echo -e "${GREEN}✓ Directories created${NC}"
}

# Build the API image
build_image() {
    echo -e "\n${YELLOW}Building Docker image...${NC}"
    docker build -f Dockerfile -t heart-disease-api:latest .
    echo -e "${GREEN}✓ Image built successfully${NC}"
}

# Start all services
start_services() {
    echo -e "\n${YELLOW}Starting all services...${NC}"
    $COMPOSE_CMD -f docker-compose.yml up -d
    echo -e "${GREEN}✓ Services started${NC}"
}

# Wait for services to be healthy
wait_for_services() {
    echo -e "\n${YELLOW}Waiting for services to be healthy...${NC}"
    
    # Wait for Elasticsearch
    echo -n "  Elasticsearch: "
    for i in {1..60}; do
        if curl -s http://localhost:9200/_cluster/health > /dev/null 2>&1; then
            echo -e "${GREEN}Ready${NC}"
            break
        fi
        sleep 2
        echo -n "."
    done
    
    # Wait for API
    echo -n "  API: "
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}Ready${NC}"
            break
        fi
        sleep 2
        echo -n "."
    done
    
    # Wait for Kibana
    echo -n "  Kibana: "
    for i in {1..60}; do
        if curl -s http://localhost:5601/api/status > /dev/null 2>&1; then
            echo -e "${GREEN}Ready${NC}"
            break
        fi
        sleep 3
        echo -n "."
    done
    
    # Wait for Grafana
    echo -n "  Grafana: "
    for i in {1..30}; do
        if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
            echo -e "${GREEN}Ready${NC}"
            break
        fi
        sleep 2
        echo -n "."
    done
}

# Print service URLs
print_urls() {
    echo -e "\n${GREEN}============================================================${NC}"
    echo -e "${GREEN}  All Services Running!${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
    echo -e "${BLUE}Service URLs:${NC}"
    echo -e "  • API:          ${GREEN}http://localhost:8000${NC}"
    echo -e "  • API Docs:     ${GREEN}http://localhost:8000/docs${NC}"
    echo -e "  • Metrics:      ${GREEN}http://localhost:8000/metrics${NC}"
    echo -e "  • MLflow:       ${GREEN}http://localhost:5000${NC}"
    echo -e "  • Kibana:       ${GREEN}http://localhost:5601${NC}"
    echo -e "  • Prometheus:   ${GREEN}http://localhost:9090${NC}"
    echo -e "  • Grafana:      ${GREEN}http://localhost:3000${NC} (admin/admin123)"
    echo -e "  • Alertmanager: ${GREEN}http://localhost:9093${NC}"
    echo ""
    echo -e "${BLUE}Quick Test:${NC}"
    echo '  curl http://localhost:8000/health'
    echo ""
    echo '  curl -X POST http://localhost:8000/predict \'
    echo '    -H "Content-Type: application/json" \'
    echo '    -d '"'"'{"data": [{"age": 55, "sex": 1, "cp": 3, "trestbps": 140, "chol": 230, "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2}]}'"'"
    echo ""
    echo -e "${YELLOW}To stop all services: $0 stop${NC}"
    echo -e "${YELLOW}To view logs: $0 logs${NC}"
}

# Stop all services
stop_services() {
    echo -e "${YELLOW}Stopping all services...${NC}"
    $COMPOSE_CMD -f docker-compose.yml down
    echo -e "${GREEN}✓ All services stopped${NC}"
}

# Show logs
show_logs() {
    $COMPOSE_CMD -f docker-compose.yml logs -f "${@:2}"
}

# Show status
show_status() {
    echo -e "${BLUE}Service Status:${NC}"
    $COMPOSE_CMD -f docker-compose.yml ps
}

# Clean up (remove volumes too)
cleanup() {
    echo -e "${YELLOW}Stopping and removing all containers and volumes...${NC}"
    $COMPOSE_CMD -f docker-compose.yml down -v
    echo -e "${GREEN}✓ Cleanup complete${NC}"
}

# Main logic
check_docker
check_docker_compose

case "${1:-start}" in
    start)
        setup_directories
        build_image
        start_services
        wait_for_services
        print_urls
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        start_services
        wait_for_services
        print_urls
        ;;
    logs)
        show_logs "$@"
        ;;
    status)
        show_status
        ;;
    clean)
        cleanup
        ;;
    build)
        build_image
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|status|clean|build}"
        echo ""
        echo "Commands:"
        echo "  start   - Build and start all services"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        echo "  logs    - Show logs (optionally: logs api, logs elasticsearch)"
        echo "  status  - Show service status"
        echo "  clean   - Stop and remove all containers and volumes"
        echo "  build   - Build the API image only"
        exit 1
        ;;
esac
