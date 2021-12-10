import os
from os.path import join as join_path
import cv2
import numpy as np
import pickle
import logging
from collections import OrderedDict,defaultdict
from app.utils import utils
from app.config.load_config import LoadJson


class Run_validation_inference:
    def __init__(self,infer_path='val.txt',yolo_config='yolov4_custom.cfg',overlay=True):
        self.json_pipeline = LoadJson()
        self.overlay = overlay
        # define model parameters
        self.darknet_path = self.json_pipeline.darknet_path
        self.weights_path = self.json_pipeline.get_weights_path()
        self.obj_path = self.json_pipeline.obj_path
        self.yolo_cfg = join_path(self.json_pipeline.dir_path,yolo_config)

        # get inference parameters
        self.iou ,self.cnf,self.outpath = self.json_pipeline.get_inferparams()

        self.val_path = os.path.join(self.json_pipeline.model_path,infer_path)

        self.transform()
        self.fit()

    def transform_metadata(self,image,metadata,overlay=True):
        img_metadata = list()
        for values in metadata:
            label_name, score, cords = values
            bboxes = utils.parse_cords(metadata,image.shape,self.json_pipeline.network_shape)
            img_metadata.append([label_name, score, bboxes[0].x1,bboxes[0].y1,bboxes[0].x2,bboxes[0].y2])
            image = utils.draw_boxes(image,bboxes)
        return image,img_metadata

    def transform(self):
        with open(self.val_path,'r+')as fd:
            val_data = fd.readlines()
        self.val_imgs = list(map(lambda x: x.strip(),val_data))
        return 0

    def fit(self):
        if os.path.isfile(join_path(self.json_pipeline.model_path,'metadata.pickle')):
            logging.warning('Results folder already present.Skipping validation process')
            return 0
        try:
            os.chdir(self.darknet_path)
            from app.utils import run_detection_via_darknet as darknet
            if not os.path.isfile(self.weights_path):
                raise("Weigths not found - Unable to run the predictions")
            self.darknet = darknet.Detector(self.weights_path, self.yolo_cfg, self.obj_path, cnf_thresh=0.5)
        except ImportError as err:
            raise (err)

        metadata_dict = defaultdict(list)
        for i, imgs_path in enumerate(self.val_imgs):
            img = cv2.imread(imgs_path)
            metadata = self.darknet.decode_detections(img)
            pred_img, img_data = self.transform_metadata(img,metadata,overlay=self.overlay)
            if self.overlay:
                utils.write_image(image=pred_img,filename=os.path.basename(imgs_path),dir_path=join_path(self.outpath,'images'))
            metadata_dict[os.path.basename(imgs_path)].append(img_data)
        # Store data (serialize)
        with open(join_path(self.outpath,'metadata.pickle'), 'wb') as handle:
            pickle.dump(metadata_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.chdir(self.json_pipeline.dir_path)
        return 0


if __name__ == '__main__':
    logging.basicConfig(filename='logs.log', level=logging.DEBUG,format="%(asctime)s:%(levelname)s: %(message)s",)
    logging.getLogger().addHandler(logging.StreamHandler())





