import os

import cv2
import numpy as np

from Dataset_Loader.DatasetLoader_ParkingView import DatasetLoader_ParkingView
from Graphical_Model_Generation.Graphical_Model_Generator_KOR import Graphical_Model_Generator_KOR
from LP_Detection import BBox, Quadrilateral
from LP_Detection.IWPOD_tf.iwpod_plate_detection_Min import find_lp_corner
from LP_Detection.IWPOD_tf.src.keras_utils import load_model_tf
from LP_Detection.VIN_LPD import load_model_VinLPD
from Utils import imread_uni


def generate_license_plate(generator, plate_number):
    img_gen = generator.make_LP(plate_number)
    img_gen = cv2.erode(img_gen, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    img_gen = cv2.resize(img_gen, None, fx=0.5, fy=0.5)
    return img_gen


def extract_N_track_features(img_gened, mask_text_area, img_front):
    img_gen_gray = cv2.cvtColor(img_gened, cv2.COLOR_BGR2GRAY)
    pt_gen = cv2.goodFeaturesToTrack(img_gen_gray, 500, 0.01, 5, mask=mask_text_area)  # feature extraction

    img_front_gray = cv2.cvtColor(img_front, cv2.COLOR_BGR2GRAY)
    img_front_gray_histeq = cv2.equalizeHist(img_front_gray)  # histogram equalization
    pt_tracked, status, err = cv2.calcOpticalFlowPyrLK(img_gen_gray, img_front_gray_histeq, pt_gen, None)  # feature tracking

    pt1 = pt_gen[status == 1].astype(np.int32)
    pt2 = pt_tracked[status == 1].astype(np.int32)
    return pt1, pt2


def calculate_text_area_coordinates(generator, shape):
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


if __name__ == '__main__':
    prefix_path = './Dataset_Loader/sample_image_label/파클'
    img_paths = [a for a in os.listdir(prefix_path) if a.endswith('.jpg')]

    loader = DatasetLoader_ParkingView(prefix_path)  # xml 읽을 준비
    d_net = load_model_VinLPD('../LP_Detection/VIN_LPD/weight')  # VIN_LPD 사용 준비
    iwpod_tf = load_model_tf('../LP_Detection/IWPOD_tf/weights/iwpod_net')  # iwpod_tf 사용 준비

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
            bb_vinlpd = BBox(d.x, d.y, d.w, d.h)
            boxes.append(bb_vinlpd)

        # iwpod_tf로 검출
        parallelograms = find_lp_corner(img, iwpod_tf)
        for _, p in enumerate(parallelograms):
            qb_iwpod = Quadrilateral(p[0], p[1], p[2], p[3])  # ex) p[0] : (342.353, 454.223)
            boxes.append(qb_iwpod)

        img_results = []
        generator = Graphical_Model_Generator_KOR('./Graphical_Model_Generation/BetaType/korean_LP', plate_type)  # 반복문 안에서 객체 생성 시 오버헤드가 발생
        for _, bb_or_qb in enumerate(boxes):
            img_gened = generate_license_plate(generator, plate_number)
            g_h, g_w = img_gened.shape[:2]
            mask_text_area = calculate_text_area_coordinates(generator, (g_h, g_w))
            img_front, mat_A = frontalization(img, bb_or_qb, g_w, g_h)
            pt1, pt2 = extract_N_track_features(img_gened, mask_text_area, img_front)
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

        k = 0
        key_in = 0
        while True:
            for i, res in enumerate(img_results):
                cv2.imshow(f'{i + 1}', res[k])
            key_in = cv2.waitKey()
            if key_in == ord('1'):
                k = 0
            elif key_in == ord('2'):
                k = 1
            else:
                break
        if key_in == 27:
            break