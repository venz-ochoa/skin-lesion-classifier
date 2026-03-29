#!/bin/bash

set -e
echo "--- Starting Skin Lesion AI Pipeline ---"
SCRIPT_DIR=$(dirname "$0")
VENV_PATH="$SCRIPT_DIR/../venv"

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Using virtual environment: $VENV_PATH"
else
    echo "Error: venv not found at $VENV_PATH."
    exit 1
fi
clear

echo "Running Data Setup..."
python3 data/get_data.py
clear
