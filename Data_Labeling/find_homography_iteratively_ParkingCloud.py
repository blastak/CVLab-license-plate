import os

import cv2
import numpy as np

from Dataset_Loader.DatasetLoader_ParkingView import DatasetLoader_ParkingView
from Graphical_Model_Generation.Graphical_Model_Generator_KOR import Graphical_Model_Generator_KOR
from LP_Detection import BBox, Quadrilateral
from LP_Detection.IWPOD_tf.iwpod_plate_detection_Min import find_lp_corner, cal_BB
from LP_Detection.IWPOD_tf.src.keras_utils import load_model_tf
from LP_Detection.VIN_LPD import load_model_VinLPD
from LP_Recognition.VIN_OCR import load_model_VinOCR
from Utils import imread_uni, save_json


def generate_license_plate(generator, plate_type, plate_number):
    img_gen = generator.make_LP(plate_number, plate_type)
    img_gen = cv2.erode(img_gen, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    img_gen = cv2.resize(img_gen, None, fx=0.5, fy=0.5)
    return img_gen


def extract_N_track_features(img_gened, mask_text_area, img_front, plate_type):
    img_gen_gray = cv2.cvtColor(img_gened, cv2.COLOR_BGR2GRAY)
    if plate_type == 'P5' or plate_type == 'P6':
        img_gen_gray = 255 - img_gen_gray
    pt_gen = cv2.goodFeaturesToTrack(img_gen_gray, 500, 0.01, 5, mask=mask_text_area)  # feature extraction

    img_front_gray = cv2.cvtColor(img_front, cv2.COLOR_BGR2GRAY)
    if plate_type == 'P5' or plate_type == 'P6':
        img_front_gray = 255 - img_front_gray
    img_front_gray_histeq = cv2.equalizeHist(img_front_gray)  # histogram equalization
    pt_tracked, status, err = cv2.calcOpticalFlowPyrLK(img_gen_gray, img_front_gray_histeq, pt_gen, None)  # feature tracking

    pt1 = pt_gen[status == 1].astype(np.int32)
    pt2 = pt_tracked[status == 1].astype(np.int32)
    return pt1, pt2


def calculate_text_area_coordinates(generator, shape, plate_type):
    cr_x = int(generator.char_xywh[0][0] * 1) - 10
    cr_y = int(generator.char_xywh[0][1] * 1) - 10
    cr_w = (generator.char_xywh[1][0] * 6 + generator.char_xywh[5][0]) * 1 + 20
    cr_h = generator.char_xywh[1][1] * 1 + 20
    if plate_type == 'P1-2':
        cr_x = int(generator.char_xywh[0][0] * 1) - 5
        cr_y = int(generator.char_xywh[0][1] * 1) - 10
        cr_w = (generator.char_xywh[1][0] * 6 + generator.char_xywh[5][0]) * 1 + 10
        cr_h = generator.char_xywh[1][1] * 1 + 20
    if plate_type == 'P1-3' or plate_type == 'P1-4' or plate_type == 'P4':
        cr_x = int(generator.char_xywh[0][0] * 1) - 10
        cr_y = int(generator.char_xywh[0][1] * 1) - 10
        cr_w = (generator.char_xywh[1][0] * 7 + generator.char_xywh[7][0]) * 1 + 20
        cr_h = generator.char_xywh[1][1] * 1 + 20
    elif plate_type == 'P3':
        cr_x = int(generator.char_xywh[6][0] * 1) - 10
        cr_y = int(generator.char_xywh[0][1] * 1) - 8
        cr_w = (generator.char_xywh[9][0] * 4 + generator.char_xywh[7][0]) * 1 + 20
        cr_h = generator.char_xywh[1][1] * 1 + generator.char_xywh[7][1] + 12 + 20
    elif plate_type == 'P5':
        cr_x = 22 - 10
        cr_y = 14 - 10
        cr_w = 296 + 20
        cr_h = 142 + 20
    elif plate_type == 'P6':
        cr_x = 15 - 10
        cr_y = 13 - 10
        cr_w = 305 + 20
        cr_h = 143 + 20
    mask_text_area = np.zeros(shape[:2], dtype=np.uint8)
    mask_text_area[cr_y:cr_y + cr_h, cr_x:cr_x + cr_w] = 255
    return mask_text_area


def frontalization(img_big, bb_or_qb, gen_w, gen_h):
    if 'Quad' in str(bb_or_qb.__class__):
        pt_src = np.float32([bb_or_qb.xy1, bb_or_qb.xy2, bb_or_qb.xy3])
    else:
        b = bb_or_qb
        pt_src = np.float32([(b.x, b.y), (b.x + b.w, b.y), (b.x + b.w, b.y + b.h)])
    pt_dst = np.float32([[0, 0], [gen_w, 0], [gen_w, gen_h]])
    mat_A = cv2.getAffineTransform(pt_src, pt_dst)
    img_front = cv2.warpAffine(img_big, mat_A, [gen_w, gen_h])  # 입력 이미지(img_big)를 gen과 같은 크기로 warping
    return img_front, mat_A


def find_homography_with_minimum_error(img_gen, mask_text_area, img_front, pt1, pt2):
    if len(pt1) < 4:
        return None
    img_front_gray = cv2.cvtColor(img_front, cv2.COLOR_BGR2GRAY)
    img_front_adap_th = cv2.adaptiveThreshold(img_front_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 0)
    mat_H = None
    minval = float('inf')
    for ransac_th in range(1, 15):
        H, stat = cv2.findHomography(pt1, pt2, cv2.RANSAC, ransacReprojThreshold=ransac_th)
        if H is not None:
            img_warped = cv2.warpPerspective(img_gen, H, img_gen.shape[1::-1])
            img_warped_gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
            _, img_warped_th = cv2.threshold(img_warped_gray, 0, 255, cv2.THRESH_OTSU)

            mask_warped = cv2.warpPerspective(mask_text_area, H, img_gen.shape[1::-1])
            img_front_adap_th_ = cv2.bitwise_and(img_front_adap_th, mask_warped)
            img_warped_th_ = cv2.bitwise_and(img_warped_th, mask_warped)
            img_subtract = cv2.absdiff(img_warped_th_, img_front_adap_th_)
            s = img_subtract.sum() / mask_warped.sum()
            if minval > s:
                minval = s
                mat_H = H.copy()
    return mat_H


def calculate_total_transformation(mat_A, mat_H):
    # Transform Matrix 역변환
    mat_A_homo = np.vstack((mat_A, [0, 0, 1]))
    mat_A_inv = np.linalg.inv(mat_A_homo)
    mat_T = mat_A_inv @ mat_H
    return mat_T


def save_quad(dst_xy, plate_type, plate_number, path, imagePath, imageHeight, imageWidth):
    shapes = []
    quad_xy = dst_xy.tolist()
    bb = cal_BB(quad_xy)
    bb_xy = [[bb[0].x, bb[0].y], [bb[0].x + bb[0].w, bb[0].y + bb[0].h]]
    shape = dict(
        label=plate_type + '_' + plate_number,
        points=quad_xy[0],
        group_id=None,
        description='',
        shape_type='polygon',
        flags={},
        mask=None
    )
    shapes.append(shape)
    shape = dict(
        label=plate_type + '_' + plate_number,
        points=bb_xy,
        group_id=None,
        description='',
        shape_type='rectangle',
        flags={},
        mask=None
    )
    shapes.append(shape)

    save_json(path, shapes, imagePath, imageHeight, imageWidth)


def cal_IOU(b, p):
    rect = np.float32([(b.x, b.y), (b.x + b.w, b.y), (b.x + b.w, b.y + b.h), (b.x, b.y + b.h)])
    para = np.float32([p[0], p[1], p[2], p[3]])
    inter_area, _ = cv2.intersectConvexConvex(rect, para)
    rect_area = b.w * b.h
    para_area = cv2.contourArea(para)
    union_area = rect_area + para_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


if __name__ == '__main__':
    prefix_path = r'./Dataset_Loader/sample_image_label/파클'
    img_paths = [a for a in os.listdir(prefix_path) if a.endswith('.jpg')]

    loader = DatasetLoader_ParkingView(prefix_path)  # xml 읽을 준비
    d_net = load_model_VinLPD('../LP_Detection/VIN_LPD/weight')  # VIN_LPD 사용 준비
    r_net = load_model_VinOCR('../LP_Recognition/VIN_OCR/weight')
    iwpod_tf = load_model_tf('../LP_Detection/IWPOD_tf/weights/iwpod_net')  # iwpod_tf 사용 준비
    error_list = []

    for _, img_path in enumerate(img_paths):
        img = imread_uni(os.path.join(prefix_path, img_path))  # 이미지 로드
        i_h, i_w = img.shape[:2]
        plate_type, plate_number, left, top, right, bottom = loader.parse_info(img_path[:-4] + '.xml')  # xml 로드

        boxes = []
        # XML 값 저장
        bb_xml = BBox(left, top, right - left, bottom - top, plate_number, int(plate_type[1:2]) - 1, 1.0)
        boxes.append(bb_xml)

        # VIN_LPD로 검출
        d_out = d_net.resize_N_forward(img)
        for _, d in enumerate(d_out):
            bb_vinlpd = BBox(d.x, d.y, d.w, d.h, class_str=d.class_str, class_idx=d.class_idx)
            boxes.append(bb_vinlpd)

        # iwpod_tf로 검출
        parallelograms = find_lp_corner(img, iwpod_tf)
        for _, p in enumerate(parallelograms):
            qb_iwpod = Quadrilateral(p[0], p[1], p[2], p[3])  # ex) p[0] : (342.353, 454.223)
            boxes.append(qb_iwpod)

        # XML BBox와 iwpod_tf corner 비교
        IOU = 0
        if parallelograms:
            IOU = cal_IOU(bb_xml, parallelograms[0])
            print(IOU)

        img_results = []
        dst_xy_list = []
        generator = Graphical_Model_Generator_KOR('./Graphical_Model_Generation/BetaType/korean_LP')  # 반복문 안에서 객체 생성 시 오버헤드가 발생
        for _, bb_or_qb in enumerate(boxes):
            img_gened = generate_license_plate(generator, plate_type, plate_number)
            g_h, g_w = img_gened.shape[:2]
            mask_text_area = calculate_text_area_coordinates(generator, (g_h, g_w), plate_type)
            # mask_text_area = generator.get_text_area((g_h, g_w), plate_type)  # example
            img_front, mat_A = frontalization(img, bb_or_qb, g_w, g_h)
            pt1, pt2 = extract_N_track_features(img_gened, mask_text_area, img_front, plate_type)
            mat_H = find_homography_with_minimum_error(img_gened, mask_text_area, img_front, pt1, pt2)
            mat_T = calculate_total_transformation(mat_A, mat_H)

            # graphical model을 전체 이미지 좌표계로 warping
            img_gen_recon = cv2.warpPerspective(img_gened, mat_T, (i_w, i_h))

            # 해당 영역 mask 생성
            img_gened_white = np.full_like(img_gened[:, :, 0], 255, dtype=np.uint8)
            mask_white = cv2.warpPerspective(img_gened_white, mat_T, (i_w, i_h))

            # 영상 합성
            img1 = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask_white))
            img2 = cv2.bitwise_and(img_gen_recon, img_gen_recon, mask=mask_white)
            img_superimposed = img1 + img2

            # 좌표 계산
            dst_xy = cv2.perspectiveTransform(np.float32([[[0, 0], [g_w, 0], [g_w, g_h], [0, g_h]]]), mat_T)
            cv2.polylines(img_superimposed, [np.int32(dst_xy)], True, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)  # quadrilateral box

            img_results.append([img_superimposed, img])
            dst_xy_list.append(dst_xy)

        print(plate_number)

        if IOU > 0.5:
            i = 2
        else:
            i = 0
            error_list.append(plate_number)
        dst_xy = dst_xy_list[i]
        cv2.imshow("img", img_results[i][0])
        cv2.waitKey(1)
        save_quad(dst_xy, plate_type, plate_number, prefix_path, img_path, img.shape[0], img.shape[1])
    with open("error_log.txt", "w", encoding="utf-8") as file:
        for error in error_list:
            file.write(error + "\n")
