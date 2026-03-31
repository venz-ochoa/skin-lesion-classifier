#!/bin/bash
rm -rf venv
python3 -m venv venv
./venv/bin/pip install --upgrade pip
chmod +x src/data_pipeline.sh
./src/data_pipeline.sh