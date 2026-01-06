#!/bin/bash

# Setup script for Heart Disease Prediction MLOps Project
# This script automates the initial setup process

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MLOps Project Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
python3 -m venv .venv

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Create necessary directories
echo -e "\n${YELLOW}Creating directories...${NC}"
mkdir -p data/raw data/processed models mlruns

# Download data
echo -e "\n${YELLOW}Downloading dataset...${NC}"
python -m src.download_data --output data/raw/heart.csv

# Process data
echo -e "\n${YELLOW}Processing dataset...${NC}"
python -m src.data --input data/raw/heart.csv --output data/processed/heart_processed.csv

# Train models
echo -e "\n${YELLOW}Training models...${NC}"
python -m src.train --data data/processed/heart_processed.csv --model-dir models --cv 5

# Run tests
echo -e "\n${YELLOW}Running tests...${NC}"
pytest tests/ -v

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\nTo activate the virtual environment:"
echo -e "  ${YELLOW}source .venv/bin/activate${NC}"

echo -e "\nTo start the API server:"
echo -e "  ${YELLOW}uvicorn src.api:app --host 0.0.0.0 --port 8000${NC}"

echo -e "\nTo run tests:"
echo -e "  ${YELLOW}pytest tests/${NC}"

echo -e "\nTo build Docker image:"
echo -e "  ${YELLOW}docker build -t heart-disease-api:latest .${NC}"

