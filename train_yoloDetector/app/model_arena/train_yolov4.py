import logging
import os
from os.path import join as join_path
from shutil import copy

import cv2
import pandas as pd

from app.config.load_config import LoadJson
from app.utils import utils


class Training:
    def __init__(self, epochs=2000, batch_size=64, retrain=True, verbose=False):
        self.json_pipeline = LoadJson()
        self.classes = self.json_pipeline.get_labels()
        self.net_shape = self.json_pipeline.network_shape
        self.batch = batch_size
        self.retrain = retrain

        self.obj_path = self.json_pipeline.obj_path
        self.custom_yolo_cfg = None
        train, val = self.json_pipeline.get_train_val_path()
        self.train_path = join_path(train, 'images')
        self.valid_path = join_path(val, 'images')


        self.transform()
        self.fit()

    def generate_csv_data(self,data_files):
        data = []
        for fname in data_files:
            img = cv2.imread(fname)
            annt_fname = fname[:-3] + 'txt'
            cls, x_min, y_min, x_max, y_max = utils.get_annotations(annt_fname, img.shape)
            label = self.json_pipeline.get_labels()[cls]
            data.append([os.path.basename(fname),label,img.shape[1],img.shape[0],x_min,y_min,x_max,y_max])
        df = pd.DataFrame(data,
            columns=['Name', 'Label', 'IMG_WIDTH', 'IMG_HEIGHT', 'xmin', 'ymin', 'xmax', 'ymax'])
        df.to_csv(join_path(self.json_pipeline.model_path, 'val.csv'))
        return 0

    def transform(self):
        logging.info("Running Train Pipeline ")
        logging.info("Creating custom yolov4 configuration")
        self.custom_yolo_cfg = utils.parse_config_file(self.json_pipeline.model_cfg, nb_classes=len(self.classes),
                                                       epochs=3000, network_size=self.net_shape, batch_size=self.batch,
                                                       out_path=self.json_pipeline.dir_path)

        if not os.path.join(self.json_pipeline.darknet_path, 'darknet'):
            raise ("Make darknet before running the training pipeline")

        if not os.path.isdir(self.json_pipeline.save_weights):
            utils.make_dir(self.json_pipeline.save_weights)

        # copy annotations files in images - Yolo needs all annotations and images in same folder
        train_files = [join_path(self.json_pipeline.data_path + '/annotations/', f[:-3] + 'txt') for f in
                       os.listdir(join_path(self.json_pipeline.model_path, 'train/images'))]
        [copy(t, join_path(self.json_pipeline.model_path, 'train/images')) for t in train_files]
        print("Total Training Files {}".format(len(train_files)))

        val_files = [join_path(join_path(self.json_pipeline.data_path + '/annotations'), f[:-3] + 'txt')
                     for f in os.listdir(join_path(self.json_pipeline.model_path, 'val/images'))]
        print("Toal Validation files {}".format(len(val_files)))
        [copy(v, join_path(self.json_pipeline.model_path, 'val/images')) for v in val_files]
        logging.info("creating csv data for train and validation files")
        val_files = [join_path(self.valid_path, imgfile) for imgfile in os.listdir(self.valid_path) if
                     '.txt' not in imgfile]
        self.generate_csv_data(val_files)
        return 0

    def fit(self):
        if self.retrain:
            train_cmd = os.path.join(self.json_pipeline.darknet_path, 'darknet') + ' detector train ' + self.obj_path + \
                        ' ' + self.custom_yolo_cfg + ' ' + self.json_pipeline.pre_trainweights + ' -dont_show -clear'

            # os.system(train_cmd)
        else:
            print("retrain the network with older weights !")
        return 0
