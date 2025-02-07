import argparse
import os

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from Graphical_Model_Generation.Graphical_Model_Generator_KOR import Graphical_Model_Generator_KOR
from LP_Recognition.VIN_OCR.VinOCR import load_model_VinOCR
from Utils import imread_uni, trans_eng2kor_v1p3


def visualize_bbox(img, r_out, char_xywh):
    for i, b in enumerate(r_out):
        cv2.rectangle(img, (b.x, b.y, b.w, b.h), (255, 255, 0), 1)  # bounding box
    for i in range(len(char_xywh) // 2):
        cv2.rectangle(img, (int(char_xywh[i * 2][0]), int(char_xywh[i * 2][1]), int(char_xywh[i * 2 + 1][0]), int(char_xywh[i * 2 + 1][1])), (255, 255, 255), 1)
    cv2.imshow("Image", img)
    cv2.waitKey()


def calculate_center(box):
    x_min, y_min, width, height = box
    center_x = x_min + (width / 2)
    center_y = y_min + (height / 2)
    return np.array([center_x, center_y])


def calculate_offset(reference_points, bbox_centers):
    offset_sum = 0
    # Hungarian Algorithm
    cost_matrix = np.zeros((len(reference_points), len(bbox_centers)))
    for i, ref_point in enumerate(reference_points):
        for j, bbox_point in enumerate(bbox_centers):
            cost_matrix[i, j] = np.linalg.norm(ref_point - bbox_point)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # print(row_ind, col_ind)
    for r, c in zip(row_ind, col_ind):
        offset_sum += cost_matrix[r, c]
    #     print(f"Reference Point {r} is matched with Bounding Box {c}")
    #     print(f"Distance: {cost_matrix[r, c]:.2f}")
    # print(offset_sum)
    return offset_sum


def save_GoodMatches(prefix_path, img_path, plate_type, threshold):
    move_path = os.path.join(prefix_path, f'GoodMatches_{plate_type}_Front_H{threshold}')
    save_path = os.path.join(prefix_path, f'GoodMatches_{plate_type}')
    if not os.path.exists(move_path):
        os.makedirs(move_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    os.rename(os.path.join(folder_path, img_path), os.path.join(move_path, img_path))
    os.rename(os.path.join(prefix_path, plate_type, img_path[6:]), os.path.join(save_path, img_path[6:]))
    os.rename(os.path.join(prefix_path, plate_type, img_path[6:-4] + '.json'), os.path.join(save_path, img_path[6:-4] + '.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='', help='Input Image folder')
    opt = parser.parse_args()

    prefix_path = opt.data
    generator = Graphical_Model_Generator_KOR()
    for folder_name in os.listdir(prefix_path):
        img_paths = []
        folder_path = os.path.join(prefix_path, folder_name)
        if os.path.isdir(folder_path) and folder_name.startswith('front_'):
            img_paths.extend(file_name for file_name in os.listdir(folder_path) if file_name.endswith('.jpg'))

        r_net = load_model_VinOCR('../LP_Recognition/VIN_OCR/weight')

        plate_type = folder_path.split('_')[-1]
        char_xywh = generator.char_xywh.get(plate_type)
        for _, img_path in tqdm(enumerate(img_paths), total=len(img_paths), desc=f'{folder_name}'):
            img = imread_uni(os.path.join(folder_path, img_path))  # 이미지 로드
            plate_number = img_path.split('_')[-1][:-4]

            r_out = r_net.forward(img)

            # Bounding Box visualize
            # visualize_bbox(img, r_out, char_xywh)

            if len(r_out) < 3:
                continue

            # bounding_box 중심점
            bbox_centers = []
            for i, b in enumerate(r_out):
                center = calculate_center([b.x, b.y, b.w, b.h])
                bbox_centers.append(center)

            list_char = r_net.check_align(r_out, int(folder_name[-1]))
            list_char_kr = trans_eng2kor_v1p3(list_char)
            print(''.join(list_char_kr))

            # 숫자 기준점
            reference_points = []
            if plate_type == 'P3' or plate_type == 'P4' or plate_type == 'P5':
                num = [1, 2, 4, 5, 6, 7]
            elif plate_type == 'P1-3' or plate_type == 'P1-4':
                num = [0, 1, 2, 4, 5, 6, 7]
            else:  # P1-1, P1-2, P2, P6
                num = [0, 1, 3, 4, 5, 6]
            for i in num:
                ref_center = calculate_center(list(map(int, char_xywh[i])))
                reference_points.append(ref_center)

            # 비교
            offset_sum = calculate_offset(reference_points, bbox_centers)

            threshold = 21
            if plate_type == 'P2':
                threshold = 18
            elif plate_type == 'P1-3':
                threshold = 24
            elif plate_type == 'P1-4':
                threshold = 26

            if offset_sum <= threshold:
                print('correct')
                save_GoodMatches(prefix_path, img_path, plate_type, threshold)
