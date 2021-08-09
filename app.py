from flask import Flask, request, jsonify
import easyocr
import onnxruntime
from inference_pipeline import Inference_engine
import cv2
import numpy as np
import base64

app = Flask(__name__)


# Add object detector models
Model_path = 'yolov4_1_3_320_320_static.onnx'
model = onnxruntime.InferenceSession(Model_path)
# add NLP Models
en_model = easyocr.Reader(['en'])
ar_model = easyocr.Reader(['ar'])
nlp_models = [en_model, ar_model]


@app.route('/')
def base_route():
    return 'API is running...'

@app.route('/process', methods=['POST'])
def process():
    img_string = request.json['img_string']
    jpg_original = base64.b64decode(img_string)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    input_img = cv2.imdecode(jpg_as_np, flags=1)
    
    # create Object instance
    model_infer = Inference_engine(input_img, model, nlp_models)
    payload = model_infer.get_licenceplate_info()

    return jsonify({ 'data': payload })


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


# if __name__ == '__main__':
#     app.run(debug=True)