#!/bin/bash

# Path to your virtual environment (replace 'my_venv' with your venv name)
VENV_PATH="./venv" 

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Run your Python script
python frcnn.py
python yolo.py
python detr.py

# Deactivate the virtual environment (optional, but good practice)
deactivate