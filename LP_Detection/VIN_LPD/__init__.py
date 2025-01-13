import cv2
import numpy as np

from LP_Detection import OcvYoloBase
from Utils import imread_uni, add_text_with_background


class VinLPD(OcvYoloBase):
    def __init__(self, _model_path, _weight_path, _classes_path):
        super().__init__(_model_path, _weight_path, _classes_path, _conf_thresh=0.24, _iou_thresh=0.3)


def load_model_VinLPD(path_base):
    if path_base[-1] != '/':
        path_base += '/'
    d_net = VinLPD(path_base + 'yolov3-feather.cfg', path_base + 'yolov3-feather.weights', path_base + 'yolov3-feather.names')
    return d_net


if __name__ == '__main__':
    d_net = load_model_VinLPD('./weight')
    img = imread_uni('../sample_image/seoulmp4_001036359jpg.jpg')
    d_out = d_net.forward(img)
    print(len(d_out))

    img_bb = img.copy()
    for i, bb in enumerate(d_out):
        cv2.rectangle(img_bb, (bb.x, bb.y, bb.w, bb.h), (255, 255, 0), 3)  # bounding box
        font_size = bb.w // 5  # magic number
        img_bb = add_text_with_background(img_bb, bb.class_str, position=(bb.x, bb.y - font_size), font_size=font_size, padding=0).astype(np.uint8)
    cv2.namedWindow('img_bb', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img_bb', tuple(map(lambda x: int(x * 0.9), (1920, 1080))))
    cv2.imshow('img_bb', img_bb)
    cv2.waitKey()
