from flask import Flask, request, jsonify
import easyocr
import onnxruntime
from src.inference_pipeline import Inference_engine
import cv2
import os
from PIL import Image
import numpy as np
import base64

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# Add object detector models
Model_path = os.path.join(BASE_PATH, 'src/model/yolov4_1_3_320_320_static.onnx')
model = onnxruntime.InferenceSession(Model_path)
# add NLP Models
en_model = easyocr.Reader(['en'])
ar_model = easyocr.Reader(['ar'])
nlp_models = [en_model, ar_model]
artifacts_path = os.path.join(BASE_PATH, 'artifacts/dump.txt')


@app.route('/')
def base_route():
    return 'API is running...'


@app.route('/base_get', methods=['GET'])
def base_get():
    return 'Running'


"""
This will infer the image if image request 
is in the form of File format.Image is saved 
temporarly in the /tmp/ folder and is used to 
infer from there.
"""


@app.route('/filestream', methods=['POST'])
def filestream():
    payload = None
    img_string = request.files['img_string']
    detector_flag = True
    # detector_flag = request.json['detector']
    # convert PIL image to RGB format
    input_img = np.array(Image.open(img_string.stream).convert('RGB'))
    # create Object instance
    model_infer = Inference_engine(input_img, model, nlp_models, detector_conf=0.5, nlp_conf=0.7, iou_thresh=0.5)
    payload = model_infer.get_licenceplate_info(run_detector=detector_flag)
    return jsonify({'data': payload})


@app.route('/predict', methods=['POST'])
def predict():
    payload = None
    img_string = request.json['img_string']
    detector_flag = request.json['detector']
    debug_flag = request.json['debug']

    jpg_original = base64.b64decode(img_string)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)

    input_img = cv2.imdecode(jpg_as_np, flags=1)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    # create Object instance
    if debug_flag:
        model_infer = Inference_engine(input_img, model, nlp_models, detector_conf=0.5, nlp_conf=0.7, iou_thresh=0.5,
                                       debug=True)
    else:
        model_infer = Inference_engine(input_img, model, nlp_models, detector_conf=0.5, nlp_conf=0.7, iou_thresh=0.5,
                                       debug=False)
    payload = model_infer.get_licenceplate_info(run_detector=detector_flag)

    return jsonify({'data': payload})


@app.errorhandler(Exception)
def handle_error(err):
    message = str(err.args[0]) or err
    status_code = 500
    response = {
        'success': False,
        'error': {
            'type': err.__class__.__name__,
            'message': message
        }
    }
    return jsonify(response), status_code


if __name__ == '__main__':
    app.run(
        debug=False,
        host='0.0.0.0',
        port=int(os.environ.get("PORT", 80)),
        threaded=True,
    )
