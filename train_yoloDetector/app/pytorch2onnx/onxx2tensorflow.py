from os.path import join,isdir
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import convert_to_constants
from app.config.load_config import LoadJson
from app.pytorch2onnx.tool.onnx2tensorflow import *
import logging
from PIL import Image
from app.utils import utils


class ONXX2TF:
    def __init__(self,onnx_model=None,output_dir=None,testimg= 'test.jpg'):
        self.json_pipeline = LoadJson()
        self.classes = self.json_pipeline.get_labels()
        self.shape = self.json_pipeline.network_shape
        self.test_img = testimg
        self.onnxmodel = onnx_model
        if onnx_model is None:
            self.onnxmodel = self.json_pipeline.data['ONNX2TF']['onnx_model']

        self.output_dir =output_dir
        if self.output_dir is None:
            self.output_dir = join(self.json_pipeline.model_path,self.json_pipeline.data['ONNX2TF']['out_path'])

        self.transform()
        self.fit()

    def load_modelGraph(self,model_dir):
        saved_model_loaded = tf.saved_model.load(model_dir, tags=[tag_constants.SERVING])
        graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        graph_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)
        return graph_func

    def transform(self):
        if float(tf.__version__[:2])<2.0:
            raise('Tensorflow version must be greater than 2.0')
        try :
            import onnx_tf
        except ImportError as err:
            logging.error("Build onnx-tensorflow:\n 1.github:https://github.com/onnx/onnx-tensorflow\n"
                          "2: Run git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow\n "
                          "run pip install -e .")
            raise()
        onnx_input_path = join(self.json_pipeline.dir_path, self.onnxmodel)
        if not isdir(self.output_dir):
            model = transform_to_tensorflow(onnx_input_path=onnx_input_path,pb_output_path= self.output_dir)

        self.graph_func = self.load_modelGraph(self.output_dir)
        # imported = tf.saved_model.load(self.output_dir)
        # self.inference_func = imported.signatures["serving_default"]
        # input_dtype = self.inference_func.inputs[0].dtype
        return 0

    def fit(self):
        img = utils.decode_image(self.test_img,shape=self.shape)
        # img = img/255.0
        img = np.expand_dims(img, axis=0)
        img = np.reshape(img,(1,3,256,256))
        imgtf = tf.convert_to_tensor(img,dtype='float32')
        #img = img.astype('float32')
        self.graph_func(imgtf)
        # tfimg = tf.convert_to_tensor(img)
        # preds = self.inference_func()
        # print(preds)
        #input_tensor = img[np.newaxis, ...].astype(np.float32)

        # output_dict = imported(img)










