import csv
from pathlib import Path

import cv2
import numpy as np

from LP_Detection.Bases import OcvYoloBase
from Utils import imread_uni, add_text_with_background


class VinLPD(OcvYoloBase):
    def __init__(self, _model_path, _weight_path, _classes_path):
        super().__init__(_model_path, _weight_path, _classes_path, _conf_thresh=0.24, _iou_thresh=0.3)


def load_model_VinLPD(path_base):
    if path_base[-1] != '/':
        path_base += '/'
    d_net = VinLPD(path_base + 'yolov3-feather.cfg', path_base + 'yolov3-feather.weights', path_base + 'yolov3-feather.names')
    return d_net


def VIN_to_csv(prefix_path):
    img_paths = [p.resolve() for p in prefix_path.iterdir() if p.suffix == '.jpg']
    d_net = load_model_VinLPD('./weight')

    for _, img_path in enumerate(img_paths):
        img = imread_uni(img_path)
        d_out = d_net.forward(img)[0]
        y = []
        for i, b in enumerate(d_out):
            y.append([b.class_str, b.x, b.y, b.x + b.w, b.y, b.x + b.w, b.y + b.h, b.x, b.y + b.h, b.conf])

        base_name = img_path.stem
        csv_filename = img_path.with_name(f"{base_name}.csv")
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in y:
                writer.writerow(row)


if __name__ == '__main__':
    d_net = load_model_VinLPD('./weight')
    img = imread_uni('../sample_image/seoulmp4_001036359jpg.jpg')
    d_out = d_net.forward(img)[0]
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

    # prefix_path = Path("../sample_image/testset")
    # VIN_to_csv(prefix_path)
