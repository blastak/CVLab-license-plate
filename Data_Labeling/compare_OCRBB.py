import os

import cv2
import numpy as np
from requests.packages import target
from tqdm import tqdm

from Dataset_Loader.DatasetLoader_ParkingView import DatasetLoader_ParkingView
from Graphical_Model_Generation.Graphical_Model_Generator_KOR import Graphical_Model_Generator_KOR
from LP_Detection import BBox, Quadrilateral
from LP_Detection.IWPOD_tf.iwpod_plate_detection_Min import find_lp_corner
from LP_Detection.IWPOD_tf.src.keras_utils import load_model_tf
from LP_Detection.VIN_LPD import load_model_VinLPD
from LP_Recognition.VIN_OCR import load_model_VinOCR
from Utils import imread_uni, bd_eng2kor_v1p3, add_text_with_background, trans_eng2kor_v1p3


extensions = ['.jpg', '.png', '.xml', '.json']

if __name__ == '__main__':
    prefix_path = r'D:\Dataset\LicensePlate\Dataset_reorganization\01_ParkingCloud'
    img_paths = []
    for folder_name in os.listdir(prefix_path):
        folder_path = os.path.join(prefix_path, folder_name)
        if os.path.isdir(folder_path) and folder_name.startswith('front_'):
            img_paths.extend(os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith('.jpg'))

    loader = DatasetLoader_ParkingView(prefix_path)  # xml 읽을 준비
    r_net = load_model_VinOCR('../LP_Recognition/VIN_OCR/weight')

    for _, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
        img = imread_uni(img_path)  # 이미지 로드
        plate_type, plate_number, left, top, right, bottom = loader.parse_json(img_path[:-4] + '.json')

        height, width = img.shape[:2]

        target_height = int(height * 1)
        target_width = int(width * 1)

        padded_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        offset_x = (target_width - width) // 2
        offset_y = (target_height - height) // 2
        padded_img[offset_y:offset_y + height, offset_x:offset_x + width] = img

        r_out = r_net.resize_N_forward(padded_img)

        for b in r_out:
            cv2.rectangle(padded_img, (b.x, b.y, b.w, b.h), (255, 255, 0), 1)  # bounding box
            font_size = b.w  # magic number
            char = bd_eng2kor_v1p3[b.class_str] if not b.class_str.isdigit() else b.class_str
            crop_resized_img = add_text_with_background(padded_img, char, position=(b.x, b.y - font_size), font_size=font_size, padding=0).astype(np.uint8)
            print(char, end='')
        list_char = r_net.check_align(r_out, int(plate_type[1]))
        list_char_kr = trans_eng2kor_v1p3(list_char)
        print(' -->', ''.join(list_char_kr))

        # 결과 출력
        cv2.imshow("Padded Image", padded_img)
        cv2.waitKey()