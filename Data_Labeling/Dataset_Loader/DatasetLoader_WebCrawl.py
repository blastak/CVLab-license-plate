import json
import os

import cv2
import numpy as np

from Utils import imread_uni, add_text_with_background


class DatasetLoader_WebCrawl:
    def __init__(self, base_path):
        self.base_path = base_path
        self.list_jpg = [f for f in os.listdir(self.base_path) if f.endswith('.jpg')]
        self.list_json = [f for f in os.listdir(self.base_path) if f.endswith('.json')]
        self.__valid = len(self.list_jpg) == len(self.list_json) != 0

    @property
    def valid(self):
        return self.__valid

    def parse_json(self, json_path):
        def value_mapping(xy_: list, spoints: list, j: int, left, top, right, bottom):
            xy_ += map(float, spoints[j])
            left = xy_[0] if xy_[0] <= left else left
            top = xy_[1] if xy_[1] <= top else top
            right = xy_[0] if xy_[0] >= right else right
            bottom = xy_[1] if xy_[1] >= bottom else bottom
            return left, top, right, bottom

        plate_number = plate_type = ''
        xy1, xy2, xy3, xy4 = [], [], [], []
        left = top = 999999
        right = bottom = 0
        filename = os.path.join(self.base_path, json_path)
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for s in data['shapes']:
                name = s['label']
                if name.startswith('P') and s['shape_type'] == 'polygon':
                    plate_type, plate_number = name.split('_')
                    for j in range(4):
                        left, top, right, bottom = value_mapping(locals()['xy%d' % (j + 1)], s['points'], j, left, top, right, bottom)

        except Exception as e:
            print(str(e))
            plate_type = 'P0'
        return plate_type, plate_number, xy1, xy2, xy3, xy4, left, top, right, bottom


if __name__ == '__main__':
    prefix = './sample_image_label/PM'

    loader = DatasetLoader_WebCrawl(prefix)

    cv2.namedWindow('img_with_label', cv2.WINDOW_NORMAL)

    jpg_paths = [a for a in os.listdir(prefix) if a.endswith('.jpg')]
    for jpg_path in jpg_paths:
        img = imread_uni(os.path.join(prefix, jpg_path))
        plate_type, plate_number, xy1, xy2, xy3, xy4, left, top, right, bottom = loader.parse_json(jpg_path[:-4] + '.json')

        img_with_label = img.copy()
        cv2.rectangle(img_with_label, (int(left), int(top)), (int(right), int(bottom)), (255, 255, 0), 3)  # bounding box
        cv2.polylines(img_with_label, [np.int32([xy1, xy2, xy3, xy4])], True, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)  # quadrilateral box
        font_size = (right - left) // 5  # magic number
        img_with_label = add_text_with_background(img_with_label, plate_number, position=(left, top - font_size), font_size=font_size, padding=0)

        cv2.resizeWindow('img_with_label', list(map(lambda x: x // 2, img_with_label.shape[1::-1])))
        cv2.imshow('img_with_label', img_with_label)
        cv2.waitKey()
