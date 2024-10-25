import cv2
import numpy as np

from src.keras_utils import detect_lp_width
from src.keras_utils import load_model
from src.utils import im2single

from LP_Detection import imread_uni


def find_lp_corner(img_orig):
    lp_threshold = 0.35  # default 0.35
    ocr_input_size = [80, 240]  # desired LP size (width x height)
    iwpod_net = load_model('./weights/iwpod_net')

    Ivehicle = img_orig
    iwh = np.array(Ivehicle.shape[1::-1], dtype=float).reshape((2, 1))

    ASPECTRATIO = 1
    WPODResolution = 480  # larger if full image is used directly
    lp_output_resolution = tuple(ocr_input_size[::-1])
    Llp, LlpImgs, _ = detect_lp_width(iwpod_net, im2single(Ivehicle), WPODResolution * ASPECTRATIO, 2 ** 4, lp_output_resolution, lp_threshold)

    xys2_list = []
    for j, img in enumerate(LlpImgs):
        pts = Llp[j].pts * iwh
        xys2 = np.transpose(pts.astype(np.int32))
        xys2_list.append(xys2.tolist())
    if xys2_list:
        return xys2_list[0]
    else:
        return [0]


def cal_center(pre_cen):
    x_coords = [point[0] for point in pre_cen]
    y_coords = [point[1] for point in pre_cen]
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    return [[min_x, min_y], [max_x, max_y]]


if __name__ == '__main__':
    img_path = "../sample_image/seoulmp4_001036359jpg.jpg"
    img = imread_uni(img_path)
    x = find_lp_corner(img)
    y = cal_center(x)
    print(x)
    print(y)

    img_bb_qb = img.copy()
    cv2.rectangle(img_bb_qb, y[0], y[1], (255, 255, 0), 3)  # bounding box
    cv2.polylines(img_bb_qb, [np.int32(x)], True, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)  # quadrilateral box
    cv2.namedWindow('img_bb_qb', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img_bb_qb', tuple(map(lambda x: int(x * 0.9), (1920, 1080))))
    cv2.imshow('img_bb_qb', img_bb_qb)
    cv2.waitKey()

    # negative test
    neg_img = np.zeros_like(img)
    x = find_lp_corner(neg_img)
    y = cal_center(x)
    print(x)
    print(y)
