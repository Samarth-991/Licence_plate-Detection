# License Plate Detection with OCR Integration

The repository uses Yolo-object detection for detecting license plates and use Easy-OCR engine to extract ocr information.

### What is this repository for?

* OCR-License plate is a Detection and Character recognition repo.The repo consists of pre-trained models for Auto Licence plate detecion and OCR model is trained specifically for Arabic and English language.OCR Plate recognition API accepts binary image and toogle argument for detector.

OCR-Model returns json data in form of list consiting of Filtered data and RAW data.The user can select depending on needs what data to be extracted.

* Run OCR-Model Standalone
OCR-Model is designed to run as standalone process as it does not require Internet.OCR-Models are present at location BASE_PATH/src/models to run the models standalone we need to copy the model files at .Easyocr/model directory

Copy all easyocr model files in easyocr dir 
* mv BASE_PATH/app/src/model/arabic.pth ~/.EasyOCR/model/
* mv BASE_PATH/app/src/model/craft_mlt_25k.pth ~/.EasyOCR/model/
* mv BASE_PATH/app/src/model/english_g2.pth ~/.EasyOCR/model

### Dependencies 
In case setup is done using conda environment use environment.yaml to create the environment at local setup.Name of the virtual environment can be change by editing the name in environment.yml
* conda env create -f environment.yml
### Deployment instructions
Deployment is done via Docker. Use Dockerfile for Deployment on local server.

* Docker creation
using docker file environment can be created with command "docker build -t ocr-model:version_name". Run the docker in the interactive mode to check the output.

### How to run tests

* Postman tool enables to POST images and get the output in json format. API provides flexibility to POST image as filestream or predict.For local test use, URL+'/filetream' else use URL+'/predict' for binary images

* Use test.py to run the test locally. Toggle between the Local Endpoint or Docker Endpoint and pass the image path on a separate terminal window.

[Results for Yolo Object Detection]
========  
------------
![Image text](https://github.com/Samarth-991/Licence_plate-Detection/blob/main/train_yoloDetector/predicted.jpg)

### Project Details 
* Version:3.6.1
* Author : samarth Tandon
* Team contact : Alok
