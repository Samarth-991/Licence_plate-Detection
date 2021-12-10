import os
from os.path import join as join_path
import json

class LoadJson:
    def __init__(self,):
        self.dir_path = os.getcwd()
        self.data = None
        self.json_file = os.path.join(self.dir_path,'pipeline.json')

        if os.path.isfile(self.json_file):
            with open(self.json_file,'r') as fd:
                self.data = json.load(fd)
            # set data path
            self.data_path = join_path(join_path(self.dir_path,self.data['Data']['data_dir']))
            # set data path used by model with/without augumentaion

            self.model_path = join_path(self.dir_path, self.data['Data']['model_data'])
            # define Image shape
            self.network_shape = self.data['Data']['network_shape']
            # define Model Parameters
            self.darknet_path = join_path(self.dir_path, self.data['Model']['Darknet_path'])
            self.model_cfg = join_path(join_path(self.darknet_path,'cfg'),self.data['Model']['yolov4_cfg'])
            self.pre_trainweights = join_path(self.darknet_path, self.data['Model']['Yolo_weights'])
            self.save_weights = join_path(self.model_path, self.data['Model']['Backup_folder'])
            # define train parametres
            self.obj_path = join_path(join_path(self.model_path,'obj'), self.data['Data']['obj_data'])
        else:
            raise ('config not found'.format(self.json_file))

    def get_train_val_path(self):
        train_path = join_path(self.model_path,'train')
        val_path = join_path(self.model_path,'val')
        return (train_path,val_path)

    def get_inferparams(self):
        iou = float(self.data['Inference']['iou_thresh'])
        cnf = float(self.data['Inference']['cnf_thesh'])
        result_path = str(join_path(self.model_path,self.data['Inference']['results']))
        return (iou,cnf,result_path)

    def get_labels(self):
        return self.data['Data']['classes']

    def get_weights_path(self):
        weights_path = os.path.join(self.save_weights,'yolov4_custom_'+self.data['Inference']['weights']+'.weights')
        if not os.path.isfile(weights_path):
            raise("Path for desired weights not found ! ")
        return weights_path








