#!/bin/bash
echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Installing project in development mode..."
pip install -e .

echo "Virtual environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate" 