#!/bin/bash

echo "--- Starting Skin Lesion AI Pipeline ---"

#detect OS and set venv path
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    VENV_PATH="./venv/Scripts"
    PYTHON_CMD="python"
else
    VENV_PATH="./venv/bin"
    PYTHON_CMD="python3"
fi

#remove old venv
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi

#create virtual environment
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv

#upgrade pip
echo "Upgrading pip..."
$VENV_PATH/python -m pip install --upgrade pip

#make the data_pipeline.sh executable
chmod +x src/data_pipeline.sh
echo "--- Running data pipeline ---"

#check if we are on windows git bash
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    #use git bash to run the script
    "C:/Program Files/Git/bin/bash.exe" src/data_pipeline.sh
else
    #linux/mac
    bash src/data_pipeline.sh
fi