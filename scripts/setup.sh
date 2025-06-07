#!/bin/bash

# Emotion Detection Project Setup Script
# This script sets up the environment and downloads necessary files

set -e  # Exit on error

echo "====================================="
echo "Emotion Detection Project Setup"
echo "====================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi
echo "✓ Python version: $python_version"

# Create virtual environment
echo -e "\nCreating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo -e "\nActivating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo -e "\nUpgrading pip..."
pip install --upgrade pip

# Install requirements
echo -e "\nInstalling requirements..."
if [ -f "requirements_minimal.txt" ]; then
    pip install -r requirements_minimal.txt
else
    pip install -r requirements.txt
fi

# Create necessary directories
echo -e "\nCreating project directories..."
mkdir -p models
mkdir -p logs
mkdir -p data
mkdir -p tests
mkdir -p ssl
echo "✓ Directories created"

# Download OpenCV haarcascade if not present
echo -e "\nChecking OpenCV data..."
python3 -c "import cv2; print('✓ OpenCV haarcascade data available')"

# Create .env file if not exists
if [ ! -f ".env" ]; then
    echo -e "\nCreating .env file..."
    cat > .env << EOF
# Environment Configuration
DEBUG=False
LOG_LEVEL=INFO
MODEL_PATH=models/fer2013_emotion_detector_final.keras
REDIS_HOST=localhost
REDIS_PORT=6379
API_HOST=0.0.0.0
API_PORT=8000
EOF
    echo "✓ .env file created"
fi

# Download sample model if needed
echo -e "\nChecking for trained models..."
if [ ! -f "models/fer2013_emotion_detector_final.keras" ] && [ ! -f "fine_tuned_best_model.keras" ]; then
    echo "⚠️  No trained model found."
    echo "Please train a model using FER2013_Improved.ipynb first"
fi

# Set up pre-commit hooks (optional)
if command -v pre-commit &> /dev/null; then
    echo -e "\nSetting up pre-commit hooks..."
    pre-commit install
    echo "✓ Pre-commit hooks installed"
fi

# Run tests
echo -e "\nRunning basic tests..."
python3 -m pytest tests/ -v --tb=short || true

echo -e "\n====================================="
echo "Setup Complete!"
echo "====================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run emotion detection: python main.py"
echo "3. Or train a new model: python main.py --train"
echo ""
echo "For image prediction:"
echo "- python predict_emotion.py /path/to/image.jpg"
echo ""