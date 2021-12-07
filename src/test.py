import requests
import base64
import os

BASE_PATH = "/home/tandonsa/PycharmProjects/side_project/ocr_mawaqif/app/notebooks"
docker_url = 'http://172.17.0.2:80/predict'
local_url = 'http://127.0.0.1:8000/predict'


def process():
    img_path = os.path.join(BASE_PATH, 'test_imgs/31.jpg')
    print("Testing {}".format(img_path))
    encodedImage = base64.b64encode(open(img_path, "rb").read()).decode()
    payload = {
        'img_string': encodedImage,
        'detector': False,
        'debug': False
    }

    response = requests.post(docker_url, json=payload)
    print(response.json())


if __name__ == '__main__':
    process()
