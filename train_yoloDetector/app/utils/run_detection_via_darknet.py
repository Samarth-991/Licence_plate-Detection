import os
from darknet import darknet as dn
import cv2
dn.set_gpu(0)


class Detector:
    def __init__(self,weights_path,yolo_cfg_path,obj_data,cnf_thresh=0.5):
        self.weights = weights_path
        self.yolo_cfg = yolo_cfg_path
        self.objdata = obj_data
        self.cnfthesh = cnf_thresh
        # load network
        self.network, self.class_names, self.colors = dn.load_network(self.yolo_cfg, self.objdata, self.weights)

    def decode_detections(self,image):
        # Darknet doesn't accept numpy images.
        # Create one with image we reuse for each detect
        self.nwidth = dn.network_width(self.network)
        self.nheight = dn.network_height(self.network)
        darknet_image = dn.make_image(self.nwidth, self.nheight, 3)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.nwidth, self.nheight),
                                   interpolation=cv2.INTER_LINEAR)

        dn.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = dn.detect_image(self.network, self.class_names, darknet_image, thresh=self.cnfthesh)
        dn.free_image(darknet_image)
        return detections

#
# if __name__ == '__main__':
#     weights_path = '/model_data/models/yolov4_custom_final.weights'
#     yolo_cfg_path = '/yolov4_custom.cfg'
#     classes_path = '/model_data/obj.names'
#     data_path = '/model_data/obj.data'
#     img_path  = '/model_data/val/images'
#     result_path = '/model_data/results'
#
#     darknetdetector = Detector(weights_path,yolo_cfg_path,data_path,cnf_thresh=0.3)
#
#     image_files = os.listdir(img_path)
#     with open("image_objects.txt", "w") as file:
#         num = 0
#         for image_file in image_files:
#             image_file = os.path.join(img_path,image_file)
#             image = cv2.imread(image_file)
#             detections = darknetdetector.image_detections(image)
#             predicted_img = darknetdetector.decode_detections(image,detections)
#             save_path = os.path.join(result_path,os.path.basename(image_file))
#             cv2.imwrite(save_path,predicted_img)
#             break
