import csv
import os
import sys

import cv2
import numpy as np

from LP_Detection.Bases import BBox
from Utils import imread_uni

sys.path.append(os.path.dirname(__file__))
from src.keras_utils import detect_lp_width, load_model_tf
from src.utils import im2single


def find_lp_corner(img_orig, iwpod_net):
    lp_threshold = 0.35  # default 0.35
    ocr_input_size = [80, 240]  # desired LP size (width x height)

    Ivehicle = img_orig
    iwh = np.array(Ivehicle.shape[1::-1], dtype=float).reshape((2, 1))

    ASPECTRATIO = 1
    WPODResolution = 480  # larger if full image is used directly
    lp_output_resolution = tuple(ocr_input_size[::-1])
    Llp, LlpImgs, _ = detect_lp_width(iwpod_net, im2single(Ivehicle), WPODResolution * ASPECTRATIO, 2 ** 4,
                                      lp_output_resolution, lp_threshold)

    xys2_list = []
    for j, img in enumerate(LlpImgs):
        pts = Llp[j].pts * iwh
        xys2 = np.transpose(pts)
        xys2_list.append(xys2.tolist())
    return xys2_list


def cal_BB(pre_cen):
    BB_list = []
    for i in range(len(pre_cen)):
        x_coords = [point[0] for point in pre_cen[i]]
        y_coords = [point[1] for point in pre_cen[i]]
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        BB_list.append(BBox(min_x, min_y, max_x - min_x, max_y - min_y))
    return BB_list


def save_csv(data, path):
    print(path)
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        # 각 4각형마다 한 줄에 8개 값(x1,y1,x2,y2,x3,y3,x4,y4) 저장
        for quad in data:
            quad_array = np.array(quad)
            # 2차원 배열을 1차원으로 평탄화
            flat_coords = quad_array.reshape(-1)
            writer.writerow(flat_coords)


def load_csv(path):
    coordinates_list = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # 문자열을 float으로 변환
            coords = np.array([float(x) for x in row])
            # 1차원 배열을 4x2 형태로 재구성
            quad_coords = coords.reshape(4, 2)
            coordinates_list.append(quad_coords)

    return coordinates_list, len(coordinates_list)


if __name__ == '__main__':
    iwpod_tf = load_model_tf('./weights/iwpod_net')

    # img_path = "../sample_image/14266136_P1-2_01루4576.jpg"
    img_path = "../sample_image/example_aolp_fullimage.jpg"
    img = imread_uni(img_path)
    x = find_lp_corner(img, iwpod_tf)
    y = cal_BB(x)
    print(x)
    print(y)

    img_bb_qb = img.copy()
    for i, b in enumerate(y):
        b.round_()
        cv2.rectangle(img_bb_qb, (b.x, b.y, b.w, b.h), (255, 255, 0), 3)  # bounding box
        cv2.polylines(img_bb_qb, [np.int32(x[i])], True, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)  # quadrilateral box
    cv2.imshow('img_bb_qb', img_bb_qb)
    cv2.waitKey()

    # negative test
    x = find_lp_corner(np.zeros_like(img), iwpod_tf)
    y = cal_BB(x)
    print(x)
    print(y)
