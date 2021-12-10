import os
import sys
from os.path import join
from app.utils import utils
from app.config.load_config import LoadJson
import splitfolders
import logging


class Create_dataset:
    def __init__(self,train_ratio=0.8, valid_ratio=0.2,img_ext='jpg'):
        self.ratio = [train_ratio, valid_ratio]
        self.img_ext = img_ext
        # load pipeline.json
        self.json_pipeline = LoadJson()

        # define variables
        self.imgsfiles= list()
        self.train_files = list()
        self.val_files = list()

        self.transform()
        self.fit()

    def transform(self):
        data_path = self.json_pipeline.data_path
        logging.info("Running Data Pipeline")
        logging.info("splitting data into train and validation from {}".format(os.path.dirname(data_path)))
        # split folders into train and valid
        if not os.path.isdir(self.json_pipeline.model_path):
            splitfolders.ratio(data_path, output=self.json_pipeline.model_path, seed=1337, ratio=(self.ratio[0], self.ratio[1]),
                               group_prefix=None)
        else:
            logging.warning('Path found. Skipping step to split dataset')
        dir_train,dir_val = self.json_pipeline.get_train_val_path()
        train_path = join(dir_train,'images')
        self.train_files = [join(train_path,imgfile) for imgfile in os.listdir(train_path)]
        val_path = join(dir_val,'images')
        self.val_files = [join(val_path,imgfile) for imgfile in os.listdir(val_path)]
        return 0

    def fit(self):
        # create train and validation txt files required by Yolo model
        utils.write_data(self.train_files,join(self.json_pipeline.model_path,'train.txt'))
        utils.write_data(self.val_files, join(self.json_pipeline.model_path, 'val.txt'))
        # create obj.names file
        obj_names = join(join(self.json_pipeline.model_path,'obj'),'obj.names')
        utils.write_data(self.json_pipeline.get_labels(),obj_names)
        # create obj-data
        obj_data = join(join(self.json_pipeline.model_path,'obj'),'obj.data')
        if not os.path.isfile(obj_data):
            with open(obj_data,'w+') as objfd:
                objfd.write("classes = {}\n".format(len(self.json_pipeline.get_labels())))
                objfd.write("train = {}\n".format(join(self.json_pipeline.model_path,'train.txt')))
                objfd.write("val = {}\n".format(join(self.json_pipeline.model_path,'val.txt')))
                objfd.write("names = {}\n".format(obj_names))
                objfd.write("backup = {}\n".format(join(self.json_pipeline.model_path,
                                                        self.json_pipeline.data['Model']['Backup_folder'])))
        logging.info("OBJ_path: {}".format(obj_data))
        return 0

if __name__ == '__main__':
    logging.basicConfig(filename='logs.log', level=logging.INFO,format="%(asctime)s:%(levelname)s: %(message)s",)
    logging.getLogger().addHandler(logging.StreamHandler())
