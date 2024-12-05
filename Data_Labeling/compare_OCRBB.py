import argparse
import os

import cv2
import numpy as np
from tensorflow.python.lib.io.file_io import rename
from tqdm import tqdm

from Graphical_Model_Generation.Graphical_Model_Generator_KOR import Graphical_Model_Generator_KOR
from LP_Recognition.VIN_OCR import load_model_VinOCR
from Utils import imread_uni, trans_eng2kor_v1p3


def calculate_center(box):
    """바운딩박스의 중심점 계산"""
    x_min, y_min, x_width, y_height = box
    center_x = x_min + (x_width / 2)
    center_y = y_min + (y_height / 2)
    return np.array([center_x, center_y])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='', help='Input Image folder')
    opt = parser.parse_args()

    prefix_path = opt.data
    generator = Graphical_Model_Generator_KOR('./Graphical_Model_Generation/BetaType/korean_LP')
    for folder_name in os.listdir(prefix_path):
        img_paths = []
        folder_path = os.path.join(prefix_path, folder_name)
        if os.path.isdir(folder_path) and folder_name.startswith('front_'):
            img_paths.extend(os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith('.jpg'))

        r_net = load_model_VinOCR('../LP_Recognition/VIN_OCR/weight')

        filtered_files = []  # 유지할 파일

        plate_type = folder_path.split('_')[-1]
        char_xywh = generator.LP_char_xywh.get(plate_type)
        reference_point = (int(char_xywh[0][0]), int(char_xywh[0][1]), int(char_xywh[1][0]), int(char_xywh[1][1]))
        reference_point = calculate_center(reference_point)
        for _, img_path in tqdm(enumerate(img_paths), total=len(img_paths), desc=f'{folder_name}'):
            img = imread_uni(img_path)  # 이미지 로드
            plate_number = img_path.split('_')[-1][:-4]

            height, width = img.shape[:2]

            target_height = int(height * 1)
            target_width = int(width * 1)

            padded_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            offset_x = (target_width - width) // 2
            offset_y = (target_height - height) // 2
            padded_img[offset_y:offset_y + height, offset_x:offset_x + width] = img

            r_out = r_net.resize_N_forward(padded_img)
            if len(r_out) < 3:
                continue
            list_char, box = r_net.check_align(r_out, int(folder_name[-1]))
            # for i, b in enumerate(box):
            #     if plate_type == 'P3' or plate_type == 'P4' or plate_type == 'P5':
            #         if i == 0:
            #             cv2.rectangle(padded_img, (b.x, b.y, b.w, b.h), (255, 255, 0), 1)  # bounding box
            #             cv2.rectangle(padded_img, (int(char_xywh[i * 2][0]), int(char_xywh[i * 2][1]), int(char_xywh[i * 2 + 1][0]), int(char_xywh[i * 2 + 1][1])), (255, 255, 255), 1)
            #         elif i > 1:
            #             cv2.rectangle(padded_img, (b.x, b.y, b.w, b.h), (255, 255, 0), 1)  # bounding box
            #             cv2.rectangle(padded_img, (int(char_xywh[(i-1) * 2][0]), int(char_xywh[(i-1) * 2][1]), int(char_xywh[(i-1) * 2 + 1][0]), int(char_xywh[(i-1) * 2 + 1][1])), (255, 255, 255), 1)
            #     else:
            #         cv2.rectangle(padded_img, (b.x, b.y, b.w, b.h), (255, 255, 0), 1)  # bounding box
            #         cv2.rectangle(padded_img, (int(char_xywh[i * 2][0]), int(char_xywh[i * 2][1]), int(char_xywh[i * 2 + 1][0]), int(char_xywh[i * 2 + 1][1])), (255, 255, 255), 1)

            list_char_kr = trans_eng2kor_v1p3(list_char)
            print(''.join(list_char_kr))

            # cv2.imshow("Padded Image", padded_img)
            # cv2.waitKey()

            threshold = 25

            if len(list_char_kr) == len(plate_number):
                offset_sum = 0
                for i, b in enumerate(box):
                    if plate_type == 'P3' or plate_type == 'P4' or plate_type == 'P5':
                        if i==0:
                            center1 = calculate_center([b.x, b.y, b.w, b.h])
                            center2 = calculate_center([box[1].x, box[1].y, box[1].w, box[1].h])
                            mid = (center1 + center2) // 2
                            reference_point = calculate_center([int(char_xywh[i * 2][0]), int(char_xywh[i * 2][1]), int(char_xywh[i * 2 + 1][0]), int(char_xywh[i * 2 + 1][1])])
                            offset_sum += np.linalg.norm(mid - reference_point)
                        elif i > 1:
                            center = calculate_center([b.x, b.y, b.w, b.h])
                            reference_point = calculate_center([int(char_xywh[(i-1) * 2][0]), int(char_xywh[(i-1) * 2][1]), int(char_xywh[(i-1) * 2 + 1][0]), int(char_xywh[(i-1) * 2 + 1][1])])
                            offset_sum += np.linalg.norm(center - reference_point)
                    else:
                        center = calculate_center([b.x, b.y, b.w, b.h])
                        reference_point = calculate_center([int(char_xywh[i * 2][0]), int(char_xywh[i * 2][1]), int(char_xywh[i * 2 + 1][0]), int(char_xywh[i * 2 + 1][1])])
                        offset_sum += np.linalg.norm(center - reference_point)
                print(offset_sum)
                if offset_sum <= threshold:  # threshold보다 가까우면 저장
                    filtered_files.append(plate_number)
                    save_path = os.path.join(prefix_path, 'good_' + plate_type)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    rename(img_path, os.path.join(save_path, img_path.split('\\')[-1]))
                print("추출 파일 목록:", len(filtered_files))
