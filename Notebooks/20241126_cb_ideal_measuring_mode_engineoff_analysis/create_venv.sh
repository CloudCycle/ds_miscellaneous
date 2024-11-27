#!/bin/bash
# Create a virtual environment
python3 -m venv .venv
# Upgrade pip
./.venv/bin/pip install --upgrade pip
# Install requirements
./.venv/bin/pip install -r ./requirements.txt
# Activate the virtual environment
source .venv/bin/activate
# Wait for 30 seconds
echo "Press any key to open Visual Studio Code"
read -n 1 -s -t 30
# Open Visual Studio Code
code .