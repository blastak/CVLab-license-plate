import os
from xml.etree.ElementTree import parse

import cv2

from Utils import imread_uni, add_text_with_background


class DatasetLoader_ParkingView:
    def __init__(self, base_path):
        self.base_path = base_path
        self.list_jpg = [f for f in os.listdir(self.base_path) if f.endswith('.jpg')]
        self.list_json = [f for f in os.listdir(self.base_path) if f.endswith('.json')]
        self.__valid = len(self.list_jpg) == len(self.list_json) != 0

    @property
    def valid(self):
        return self.__valid

    def parse_info(self, xml_path):
        plate_number = plate_type = ''
        left = top = right = bottom = 0
        try:
            xml_tree = parse(os.path.join(self.base_path, xml_path))
            xml_root = xml_tree.getroot()
            xml_objects = xml_root.findall('object')
            names = [obj.findtext('name') for obj in xml_objects]
            for i, name in enumerate(names):
                if name.find('P') != -1:
                    plate_type, plate_number = name.split('_')
                    xml_bndbox = xml_objects[i].find('bndbox')
                    left = int(xml_bndbox.findtext('xmin'))
                    top = int(xml_bndbox.findtext('ymin'))
                    right = int(xml_bndbox.findtext('xmax'))
                    bottom = int(xml_bndbox.findtext('ymax'))
        except Exception as e:
            print(str(e))
            plate_type = 'P0'
            plate_number = xml_path[:-4].split(sep='_')[-1]
        plate_type = 'P1-1' if plate_type == 'P1' else plate_type  # P1 일 경우 P1-1 로 변경
        return plate_type, plate_number, left, top, right, bottom


if __name__ == '__main__':
    prefix = './sample_image_label/파클'
    # prefix = './sample_image_label/충남'
    # prefix = './sample_image_label/용산'

    loader = DatasetLoader_ParkingView(prefix)

    jpg_paths = [a for a in os.listdir(prefix) if a.endswith('.jpg')]
    for jpg_path in jpg_paths:
        img = imread_uni(os.path.join(prefix, jpg_path))
        plate_type, plate_number, left, top, right, bottom = loader.parse_info(jpg_path[:-4] + '.xml')

        img_with_label = img.copy()
        cv2.rectangle(img_with_label, (left, top), (right, bottom), (255, 255, 0), 3)  # bounding box
        font_size = (right - left) // 5  # magic number
        img_with_label = add_text_with_background(img_with_label, plate_number, position=(left, top - font_size), font_size=font_size, padding=0)
        cv2.imshow('img_with_label', img_with_label)
        cv2.waitKey()
