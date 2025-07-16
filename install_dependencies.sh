#!/bin/bash

echo "Installing dependencies for LLM project..."

# Install pip if not available
if ! command -v pip3 &> /dev/null; then
    echo "pip3 not found. Installing pip..."
    sudo apt update
    sudo apt install -y python3-pip
fi

# Install TensorFlow and numpy
echo "Installing TensorFlow..."
pip3 install tensorflow

echo "Installing numpy..."
pip3 install numpy

echo "Dependencies installed successfully!"
echo "You can now run: python3 llm_tensorflow.py"
