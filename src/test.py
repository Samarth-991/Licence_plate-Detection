import requests
import base64 
import os 
url  = 'http://127.0.0.1:5000/predict'

def process():
    img_path_1 = 'IMG-20210610-WA0061.jpg'
    encodedImage = base64.b64encode(open(img_path_1, "rb").read()).decode()

    payload = {
        'img_string': encodedImage
    }

    response = requests.post(url, json=payload)

    #print(response.status_code)
    print(response.json())


process()
