import logging
import os
import pickle
import re

import cv2
import numpy as np
from imgaug.augmentables.bbs import BoundingBox
from tensorflow.keras.preprocessing import image


def make_dir(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as err:
            logging.info(err)
    return 0


def write_image(image, filename, dir_path):
    if not os.path.isdir(dir_path):
        make_dir(dir_path)
    cv2.imwrite(os.path.join(dir_path, filename), image)


def read_and_save_ascsv(metadata_path, img_names, csv_path='results.csv'):
    with open(csv_path, 'w+') as csv:
        csv.writelines("{},{},{},{},{},{},{}\n".format('image_name', 'label', 'score', 'xmin', 'ymin', 'xmax', 'ymax'))
        metadata = deserilize_pkl(metadata_path)
        for img_name in img_names:
            for values in metadata[os.path.basename(img_name)][0]:
                csv.writelines(
                    "{},{},{},{},{},{},{}\n".format(img_name, values[0], values[1], str(values[2]), str(values[3]),
                                                    str(values[2] + values[4]), str(values[3] + values[5])))
    return 0


def deserilize_pkl(pickle_path):
    if os.path.isfile(pickle_path):
        # Load data (deserialize)
        with open(pickle_path, 'rb') as handle:
            unserialized_data = pickle.load(handle)
    else:
        raise ('Serialized data not found !')
    return unserialized_data


def write_data(files_list, out_file):
    if not os.path.isdir(os.path.dirname(out_file)):
        make_dir(os.path.dirname(out_file))
    with open(out_file, 'w+') as outfd:
        for fname in files_list:
            if '.txt' not in fname:
                outfd.write("{}\n".format(fname))
    return 0


def get_annotations(annotation_file, img_shape):
    with open(annotation_file, 'r') as fd:
        cords = [cords.strip().split(' ') for cords in fd.readlines()][0]
    fd.close()
    cords = np.float_(cords)
    cls, x, y, w, h = cords
    x_min = int(float(x - w / 2) * img_shape[1])
    y_min = int(float(y - h / 2) * img_shape[0])
    x_max = int(float(x + w / 2) * img_shape[1])
    y_max = int(float(y + h / 2) * img_shape[0])
    return [int(cls), x_min, y_min, x_max, y_max]


def decode_image(img_path, shape=None, type=None):
    if shape is None:
        img = image.load_img(img_path)
    else:
        img = image.load_img(img_path, target_size=shape)
    img = image.img_to_array(img)
    return img.astype(np.float32)


def parse_cords(metadata, img_shape, network_shape):
    boxes = []
    for values in metadata:
        label_name, score, cords = values
        cords = list(map(lambda x: float(x) / network_shape[0], cords))
        x, y, w, h = cords
        x_min = int(float(x - w / 2) * img_shape[1])
        y_min = int(float(y - h / 2) * img_shape[0])
        x_max = int(float(x + w / 2) * img_shape[1])
        y_max = int(float(y + h / 2) * img_shape[0])
        boxes.append(BoundingBox(x_min, y_min, x_max, y_max, label=label_name))
    return boxes


def parse_config_file(yolo_cfg, nb_classes, network_size=(416, 416), batch_size=64, epochs=5000, steps=(4000, 4500),
                      out_path='default.cfg'):
    with open(yolo_cfg, 'r+') as yfd:
        configs = list(map(lambda x: x.strip(), yfd.readlines()))
    for i, val in enumerate(configs):
        if re.search('batch', val):
            configs[i] = configs[i].replace(val.split('=')[1], str(batch_size))
        if re.search('max_batches', val):
            configs[i] = configs[i].replace(configs[i].split('=')[1], str(epochs))
        if re.search('steps=', val):
            configs[i] = configs[i].replace(configs[i].split('=')[1], str(steps[0]) + ',' + str(steps[1]))
        if re.search('width', val):
            configs[i] = configs[i].replace(val.split('=')[1], str(network_size[0]))
        if re.search('height', val):
            configs[i] = configs[i].replace(val.split('=')[1], str(network_size[1]))
        if re.search('mosaic', val):
            configs[i] = configs[i].replace(val.split('=')[1], str(0))
        if re.search('classes', val):
            configs[i] = configs[i].replace(val.split('=')[1], str(nb_classes))
        if re.search('yolo', val):
            configs[i - 4] = configs[i - 4].replace(configs[i - 4].split('=')[1], str(3 * (nb_classes + 5)))

    new_cfg = os.path.join(out_path, 'yolov4_custom.cfg')
    with open(new_cfg, 'w+') as nfd:
        for cfg_val in configs:
            nfd.writelines("{}\n".format(cfg_val))
    nfd.close()
    return new_cfg


def draw_boxes(img, img_bboxes, color=(138, 43, 226)):
    if img_bboxes is not None:
        for box in img_bboxes:
            x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    return img


def class_colors(names):
    """
    Create a dict with one random BGR color for each
    class name
    """
    return {name: (
        np.random.randint(0, 255),
        np.random.randint(0, 255),
        np.random.randint(0, 255)) for name in names}


def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    img = np.copy(img)
    color = class_colors(class_names)

    width = img.shape[1]
    height = img.shape[0]

    for i in range(len(boxes[0])):
        box = boxes[0]
        x1 = int(box[i][0] * width)
        y1 = int(box[i][1] * height)
        x2 = int(box[i][2] * width)
        y2 = int(box[i][3] * height)
        if class_names:
            cls_conf = box[i][5]
            cls_id = box[i][6]
            cls_label = class_names[cls_id]
            img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, color[cls_label], 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color[cls_label], 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img
