import cv2
import os
import numpy as np
import onnxruntime
from app.config.load_config import LoadJson
from app.pytorch2onnx.tool.darknet2onnx import *
from app.pytorch2onnx.tool.utils import post_processing
from app.utils.utils import plot_boxes_cv2


class YOLO2ONXX:
    def __init__(self,batch_size=32,yolo_config='yolov4_custom.cfg',testimg='test.img'):
        self.json_pipeline = LoadJson()
        self.classes = self.json_pipeline.get_labels()
        # define weights and Yolo cfg
        self.weights_file = self.json_pipeline.get_weights_path()
        self.yolo_cfg= os.path.join(self.json_pipeline.dir_path,yolo_config)
        self.batch_size = batch_size
        self.testimg = testimg
        self.onnx_path =None
        self.transform()
        self.fit()

    def detect(self,img_src,img_shape,model):
        resized = cv2.resize(img_src, img_shape, interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        
	# Compute
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: img_in})
        boxes = post_processing(img_in, 0.4, 0.6, outputs)
        return boxes

    def transform(self):
        # Transform to onnx
        self.onnx_path = transform_to_onnx(self.yolo_cfg, self.weights_file, 1)
        session = onnxruntime.InferenceSession(self.onnx_path)
        # session = onnx.load(onnx_path)
        print("ONXX model loaded successfully")
        print("The model expects input shape: ", session.get_inputs()[0].shape)

    def fit(self):
        session = onnxruntime.InferenceSession(self.onnx_path)
        IN_IMAGE_H = session.get_inputs()[0].shape[2]
        IN_IMAGE_W = session.get_inputs()[0].shape[3]
        if os.path.isfile(self.testimg):
            image = cv2.imread(self.testimg)
            boxes = self.detect(image,img_shape=(IN_IMAGE_H,IN_IMAGE_W),model=session)
            plot_boxes_cv2(image,boxes,class_names=self.classes,savename='predicted.jpg')
        return 0
