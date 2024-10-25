import cv2

from LP_Detection import OcvYoloBase
from Utils import imread_uni


class VinLPD(OcvYoloBase):
    def __init__(self, _model_path, _weight_path, _classes_path):
        super().__init__(_model_path, _weight_path, _classes_path, _conf_thresh=0.24, _iou_thresh=0.5)


if __name__ == '__main__':
    d_net = VinLPD('./weight/yolov3-feather.cfg', './weight/yolov3-feather.weights', './weight/yolov3-feather.names')
    img = imread_uni('../sample_image/seoulmp4_001036359jpg.jpg')
    d_out = d_net.forward(img)
    print(len(d_out))

    img_bb = img.copy()
    for i,bb in enumerate(d_out):
        cv2.rectangle(img_bb, (bb.x, bb.y, bb.w, bb.h), (255, 255, 0), 3)  # bounding box
    cv2.namedWindow('img_bb',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img_bb', tuple(map(lambda x: int(x * 0.9), (1920, 1080))))
    cv2.imshow('img_bb', img_bb)
    cv2.waitKey()
