#!/bin/bash

# Red Block Detection Setup and Run Script

echo "Setting up Red Block Detection Environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Make the Python script executable
chmod +x cv_cam.py

echo "Setup complete!"
echo ""
echo "To run the red block detection:"
echo "  source venv/bin/activate"
echo "  python3 cv_cam.py"
echo "Or run this script again to start detection:"
echo "  ./run.sh run"

# Check if user wants to run immediately
if [ "$1" = "run" ]; then
    echo "Starting red block detection..."
    python cv_cam.py
fi