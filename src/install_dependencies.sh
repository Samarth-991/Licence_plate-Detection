#!/bin/bash

conda create -n ocr_model python=3.6 -y
conda activate ocr_model

python -m pip install --upgrade pip
pip install easyocr onnxruntime requests
pip install setuptools --upgrade
pip install flask==1.1.2 flask-socketio==4.2.1 python-engineio==3.13.2 python-socketio==4.6.1
conda env export -f environment.yml
echo "env ocr_model created. Use conda activate ocr_model to activate"
