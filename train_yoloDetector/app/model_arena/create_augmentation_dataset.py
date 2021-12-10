import os
import numpy as np
from tqdm import tqdm
import imageio
from shutil import copy
from app.utils import utils
import imgaug as ia
import logging
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from app.config.load_config import LoadJson
ia.seed(1)


class Augmentation:
    def __init__(self,augment_on='train.txt',agument=False,brightness=(0.7,1.3),emboss=(0.2,0.5),contrast=(1.2,1.4),
                 scale=False,rotate=False,add_noise=False, visualize=False):
        self.json_pipeline = LoadJson()
        self.train_path = os.path.join(self.json_pipeline.model_path,augment_on)
        # set flags
        self.agument = agument
        self.visualize = visualize
        # define variables
        self.imgsfiles= list()
        self.img_annotations = list()

        logging.info("Augmenting images on training dataset //..")
        self.transform()
        self.fit()

    def add_augimages(self,path):
        with open(self.train_path,'a+') as tp:
            tp.write("{}\n".format(path))

    def transform(self):
        with open(self.train_path,'r+')as trainfd:
            self.imgsfiles = [imgfile.strip() for imgfile in trainfd.readlines()]
        if self.visualize:
            img_file = self.imgsfiles[np.random.randint(0,len(self.imgsfiles))]
            img = utils.decode_image(img_file)
            annotation_file = img_file.replace('images', 'annotations')[:-4] + '.txt'
            bboxes = utils.get_annotations(img, annotation_file,self.json_pipeline.get_labels())  # for one single image
            annotated_img = utils.draw_boxes(img,bboxes)
            imageio.imwrite('annotated_image.jpg',annotated_img)

    def fit(self):
        agumentation_ops = iaa.Sequential([
            iaa.Multiply((0.9, 1.2)),  # change brightness, doesn't affect BBs
            iaa.LinearContrast((1.0,1.3)),
            iaa.Emboss(0.2,0.5)
            ])
        if self.agument:
            for i, imgfile in tqdm(enumerate(self.imgsfiles)):
                img = utils.decode_image(imgfile)
                annotation_file = imgfile.replace('images','annotations')[:-4]+'.txt'

                # Get bounding box aanotations
                bboxes = utils.get_annotations(img,annotation_file,self.json_pipeline.get_labels()) # for one single image
                bbs = BoundingBoxesOnImage(bboxes,shape=img.shape)
                # Augment BBs and images
                image_aug, bbs_aug = agumentation_ops(image=img, bounding_boxes=bbs)
                augmented_path = imgfile[:-4]+'_aug'+str(i)+'.jpg'
                # add augmented images path in the train.txt file
                self.add_augimages(augmented_path)
                # save augmented images in training path
                image_aug = image_aug.astype(np.uint8)
                imageio.imwrite(augmented_path,image_aug)
                annotatation_path = annotation_file[:-4] + '_aug' + str(i) + '.txt'
                copy(annotation_file,annotatation_path)
        else:
            logging.warning("Augmentation Flag off - Skipping agumentation pipeline")
        return 0