#!/bin/bash
set -e

echo "--- Starting Skin Lesion AI Pipeline ---"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

#detect OS and set venv path
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    VENV_PATH="$SCRIPT_DIR/../venv"
    PYTHON="$VENV_PATH/Scripts/python.exe"
    ACTIVATE="$VENV_PATH/Scripts/activate"
else
    VENV_PATH="$SCRIPT_DIR/../venv"
    PYTHON="$VENV_PATH/bin/python3"
    ACTIVATE="$VENV_PATH/bin/activate"
fi

#check if venv exists
if [ -d "$VENV_PATH" ]; then
    #activate virtual environment
    source "$ACTIVATE"
    echo "Using virtual environment: $VENV_PATH"
else
    echo "Error: venv not found at $VENV_PATH."
    exit 1
fi

#clear screen (works in git bash/linux)
if command -v clear >/dev/null 2>&1; then
    clear
fi

echo "Running Data Setup..."
$PYTHON "$SCRIPT_DIR/../data/get_data.py"

#clear screen again
if command -v clear >/dev/null 2>&1; then
    clear
fi

echo "--- Data Pipeline Finished ---"