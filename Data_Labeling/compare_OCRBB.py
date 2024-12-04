import os

import cv2
import numpy as np
from tqdm import tqdm

from LP_Recognition.VIN_OCR import load_model_VinOCR
from Utils import imread_uni, trans_eng2kor_v1p3

if __name__ == '__main__':
    prefix_path = r'D:\Dataset\LicensePlate\Dataset_reorganization\01_ParkingCloud'
    for folder_name in os.listdir(prefix_path):
        img_paths = []
        folder_path = os.path.join(prefix_path, folder_name)
        if os.path.isdir(folder_path) and folder_name.startswith('front_'):
            img_paths.extend(os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith('.jpg'))

        r_net = load_model_VinOCR('../LP_Recognition/VIN_OCR/weight')

        for _, img_path in tqdm(enumerate(img_paths), total=len(img_paths), desc=f'{folder_name}'):
            img = imread_uni(img_path)  # 이미지 로드

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
            list_char = r_net.check_align(r_out, int(folder_name[-1]))
            list_char_kr = trans_eng2kor_v1p3(list_char)
            print(''.join(list_char_kr))

            # 결과 출력
            cv2.imshow("Padded Image", padded_img)
            cv2.waitKey()
