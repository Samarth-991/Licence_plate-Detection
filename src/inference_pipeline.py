import logging
import numpy as np
import cv2
import logging as log
import string
import difflib


class Inference_engine:
    def __init__(self, input_image, detector_model, nlp_model, detector_conf=0.1, nlp_conf=0.4, iou_thresh=0.5):
        self.input_img = input_image
        self.input_img_width = self.input_img.shape[1]
        self.input_img_height = self.input_img.shape[0]
        # Define Prediction Cofficents
        self.detector_conf = detector_conf
        self.iou_thresh = iou_thresh
        self.nlp_conf = nlp_conf

        # flag for detection
        self.success_detection = False
        self.txt_data = None

        # Load the model once in the memory
        self.session = detector_model
        self.en_reader = nlp_model[0]
        self.ar_reader = nlp_model[1]

    def get_licenceplate_info(self):
        IN_IMAGE_H = self.session.get_inputs()[0].shape[2]
        IN_IMAGE_W = self.session.get_inputs()[0].shape[3]

        decoded_img = self.decode_img(self.input_img, shape=(IN_IMAGE_H, IN_IMAGE_W))
        detections = self.detect(decoded_img)
        boxes = self.post_processing(detections, conf_thresh=self.detector_conf,
                                     nms_thresh=self.iou_thresh)
        self.bounding_cords = self.decode_boxes(boxes)
        if self.bounding_cords is None:
            logging.info("No Detections from model")

        elif not self.check_out_of_bounds():
            cropped_alpr = self.input_img[self.bounding_cords[1]:self.bounding_cords[3],
                           self.bounding_cords[0]:self.bounding_cords[2]]
            #             lisc_plate_img = self.enhance_image(cropped_alpr.copy())
            self.txt_data = self.NLP_model(cropped_alpr.copy(), nlp_confidence=self.nlp_conf)
            print("final string", self.txt_data)
        return self.txt_data

    def enhance_image(self, crop_image, alpha=1.5, beta=0):
        gray_image = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY)
        blur_img = cv2.GaussianBlur(gray_image, (5, 5), 0)
        adpt_img = cv2.adaptiveThreshold(blur_img, 200, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 3)
        return adpt_img

    def check_out_of_bounds(self):
        out_of_bounds = False
        if (self.bounding_cords[0] > self.input_img_width) and (self.bounding_cords[2] > self.input_img_width) and (
                self.bounding_cords[1] > self.input_img_height) and (self.bounding_cords[3] > self.input_img_height):
            out_of_bounds = True
        return out_of_bounds

    def evaluate_text(self, results_en, results_ar):
        ocr_data = []
        text_string = []
        num_data = []
        nval = None
        # get results
        text_en = [r[1] for r in results_en]
        digt_en = [r[1][:5] for r in results_en if r[1].isdigit() and len(r[1]) > 4]
        # find Single Char
        single_chars = [s for s in results_en if len(s[1]) == 1]
        if len(single_chars) > 0:
            singlchar_array = np.asarray(single_chars)
            nval = singlchar_array[np.argmax(singlchar_array[:, 0])][1]
        # find numeric data from english text
        digits = [txt[:5] for txt in text_en if txt[:5].isdigit() and len(txt) > 4]
        num_data = list(set(digits).intersection(digt_en))
        if len(num_data) == 0:
            num_data = digits
        num_data.sort(reverse=True)
        # evaluate Arabic text
        text_ar = [r[1].translate({ord(i): None for i in "':!?+|\/}{*%&#()$-_=[]^., "}) for r in results_ar if
                   not any(c.isdigit() for c in r[1])]
        # find closest match to DUBAI
        if difflib.get_close_matches('DUBAI', text_en) or any('DUBAI' in word for word in text_en):
            text_string.append('DUBAI')
        # find closest match to UAE
        elif difflib.get_close_matches('UAE', text_en) or any('UAE' in word for word in text_en):
            text_string.append('UAE')
        # find closest match to AD
        elif difflib.get_close_matches('AD', text_en) or any('AD' in word for word in text_en):
            text_string.append("AD")
        ocr_data = num_data + text_string
        if nval is not None:
            ocr_data.insert(0, nval)
        ocr_data = list(ocr_data + list(set(text_ar) - set(ocr_data)))
        return ocr_data

    def NLP_model(self, cropped_img, nlp_confidence=0.0):
        en_meta_data = []
        # run NLP model on cropped image
        results_en = self.en_reader.readtext(cropped_img, allowlist=string.digits + string.ascii_uppercase)
        for rlt in results_en:
            en_meta_data.append([rlt[-1], rlt[-2]])
        results_ar = self.ar_reader.readtext(cropped_img)
        nlp_data = self.evaluate_text(en_meta_data, results_ar)
        nlp_data = list(filter(str.strip, nlp_data))
        return nlp_data

    def detect(self, decoded_image):
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.get_outputs()
        output_names = list(map(lambda output: output.name, outputs))
        detections = self.session.run(output_names, {input_name: decoded_image})
        return detections

    def nms_cpu(self, boxes, confs, nms_thresh=0.4, min_mode=False):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = confs.argsort()[::-1]
        keep = []
        while order.size > 0:
            idx_self = order[0]
            idx_other = order[1:]
            keep.append(idx_self)
            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            if min_mode:
                over = inter / np.minimum(areas[order[0]], areas[order[1:]])
            else:
                over = inter / (areas[order[0]] + areas[order[1:]] - inter)
            inds = np.where(over <= nms_thresh)[0]
            order = order[inds + 1]
        return np.array(keep)

    def post_processing(self, output, conf_thresh=0.3, nms_thresh=0.5):
        # [batch, num, 1, 4]
        box_array = output[0]
        # [batch, num, num_classes]
        confs = output[1]

        if type(box_array).__name__ != 'ndarray':
            box_array = box_array.cpu().detach().numpy()
            confs = confs.cpu().detach().numpy()
        num_classes = confs.shape[2]
        # [batch, num, 4]
        box_array = box_array[:, :, 0]
        # [batch, num, num_classes] --> [batch, num]
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)
        bboxes_batch = []
        for i in range(box_array.shape[0]):
            argwhere = max_conf[i] > conf_thresh
            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]
            l_max_id = max_id[i, argwhere]
            bboxes = []
            # nms for each class
            for j in range(num_classes):
                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]
                ll_max_id = l_max_id[cls_argwhere]
                keep = self.nms_cpu(ll_box_array, ll_max_conf, nms_thresh)
                if (keep.size > 0):
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]
                    for k in range(ll_box_array.shape[0]):
                        bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3],
                                       ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])
            bboxes_batch.append(bboxes)
        return bboxes_batch

    def decode_boxes(self, boxes):
        cords = None
        for i in range(len(boxes[0])):
            box = boxes[0]
            x1 = int(box[i][0] * self.input_img_width)
            y1 = int(box[i][1] * self.input_img_height)
            x2 = int(box[i][2] * self.input_img_width)
            y2 = int(box[i][3] * self.input_img_height)
            cords = (x1, y1, x2, y2)
        return cords

    @staticmethod
    def decode_img(img, shape=(320, 320), channel=3):
        output_img = None
        try:
            resized = cv2.resize(img, shape, interpolation=cv2.INTER_LINEAR)
            trp_img = np.transpose(resized, (2, 0, 1)).astype(np.float32)
            output_img = np.expand_dims(trp_img, axis=0)
            output_img /= 255.0
        except IOError as e:
            log.error('{}! Unable to read image'.format(e))
        return output_img
