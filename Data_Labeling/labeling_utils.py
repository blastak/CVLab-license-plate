import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from LP_Detection.Bases import BBox, Quadrilateral
from LP_Detection.IWPOD_tf.iwpod_plate_detection_Min import find_lp_corner
from Utils import imwrite_uni, save_json, trans_eng2kor_v1p3


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


def calculate_text_area_coordinates(generator, plate_type):
    number_area = generator.get_plate_number_area_only(plate_type)
    if plate_type == 'P1-2':
        margin = [-5, -10, 5, 10]
    elif plate_type == 'P3':
        margin = [-10, -8, 10, 10]
    elif plate_type in ['P5', 'P6']:
        margin = [0, 0, 0, 0]
    else:
        margin = [-10, -10, 10, 10]
    min_x, min_y, max_x, max_y = map(int, [a + m for a, m in zip(number_area, margin)])

    mask_text_area = np.zeros(generator.plate_wh[plate_type][::-1], dtype=np.uint8)
    mask_text_area[min_y:max_y, min_x:max_x] = 255
    return mask_text_area


def frontalization(img_big, bb_or_qb, gen_w, gen_h, mode=3):
    if mode == 3:
        if 'Quad' in str(bb_or_qb.__class__):
            pt_src = np.float32([bb_or_qb.xy1, bb_or_qb.xy2, bb_or_qb.xy3])
        else:
            b = bb_or_qb
            pt_src = np.float32([(b.x, b.y), (b.x + b.w, b.y), (b.x + b.w, b.y + b.h)])
        pt_dst = np.float32([[0, 0], [gen_w, 0], [gen_w, gen_h]])
        mat_T = cv2.getAffineTransform(pt_src, pt_dst)
        img_front = cv2.warpAffine(img_big, mat_T, [gen_w, gen_h])  # 입력 이미지(img_big)를 gen과 같은 크기로 warping
    else:
        if 'Quad' in str(bb_or_qb.__class__):
            pt_src = np.float32([bb_or_qb.xy1, bb_or_qb.xy2, bb_or_qb.xy3, bb_or_qb.xy4])
            pt_dst = np.float32([[0, 0], [gen_w, 0], [gen_w, gen_h], [0, gen_h]])
            mat_T = cv2.getPerspectiveTransform(pt_src, pt_dst)
            img_front = cv2.warpPerspective(img_big, mat_T, [gen_w, gen_h])
        else:
            raise NotImplementedError
    return img_front, mat_T


def find_total_transformation(img_gened, generator, plate_type, img, bb_or_qb):
    g_h, g_w = img_gened.shape[:2]
    mask_text_area = calculate_text_area_coordinates(generator, plate_type)
    img_front, mat_A = frontalization(img, bb_or_qb, g_w, g_h)
    pt1, pt2 = extract_N_track_features(img_gened, mask_text_area, img_front, plate_type)
    mat_H, mat_T = find_homography_all(pt1, pt2, mat_A)
    return mat_T


def find_homography_all(pt1, pt2, mat_A):
    mat_H = []
    mat_T = []
    for ransac_th in range(1, 15):
        H, stat = cv2.findHomography(pt1, pt2, cv2.RANSAC, ransacReprojThreshold=ransac_th)
        mat_H.append(H)
        mat_T.append(calculate_total_transformation(mat_A, H))
    return mat_H, mat_T


def calculate_total_transformation(mat_A, mat_H):
    # Transform Matrix 역변환
    mat_A_homo = np.vstack((mat_A, [0, 0, 1]))
    mat_A_inv = np.linalg.inv(mat_A_homo)
    mat_T = mat_A_inv @ mat_H
    return mat_T


def save(dst_xy_list, plate_type, plate_number, path, imagePath, imageHeight, imageWidth):
    shapes = []
    print("1 or 2 or 3")
    key_in = cv2.waitKey()
    if key_in == ord('1'):
        print(1)
        quad_xy = dst_xy_list[0]
    elif key_in == ord('2'):
        print(2)
        quad_xy = dst_xy_list[1]
    elif key_in == ord('3'):
        print(3)
        quad_xy = dst_xy_list[2]
    quad_xy = quad_xy[0].tolist()
    shapes = [
        dict(
            label=plate_type + '_' + plate_number,
            points=quad_xy,
            group_id=None,
            description='',
            shape_type='polygon',
            flags={},
            mask=None
        )
    ]
    save_json(path, shapes, imagePath, imageHeight, imageWidth)


def front_image(dst_xy, img, plate_type, img_path, prefix_path, generator, index, save=True):
    dst_quad = Quadrilateral(dst_xy[0][0], dst_xy[0][1], dst_xy[0][2], dst_xy[0][3])
    img_front, mat_A = frontalization(img, dst_quad, *generator.plate_wh[plate_type], 4)

    if save and prefix_path:
        save_dir = prefix_path / f'front_{plate_type}'
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = f'front_{img_path.stem}_{index}.jpg'  # e.g., front_img001_0.jpg
        imwrite_uni(save_dir / save_name, img_front)
    return img_front


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
    for r, c in zip(row_ind, col_ind):
        offset_sum += cost_matrix[r, c]
    return offset_sum


def compare_ocrbb(img_front, generator, plate_type, r_net):
    char_xywh = generator.char_xywh.get(plate_type)
    r_outs = r_net.forward(img_front)
    if len(r_outs[0]) < 3:
        return 100

    # bounding_box 중심점
    ocrbb_centers = []
    for i, r_out in enumerate(r_outs):
        for i, b in enumerate(r_out):
            center = calculate_center([b.x, b.y, b.w, b.h])
            ocrbb_centers.append(center)
        list_char = r_net.check_align(r_out, int(plate_type[1]))
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

    offset_sum = calculate_offset(reference_points, ocrbb_centers)

    return offset_sum


def detect_all(img, d_net, iwpod_tf):
    boxes = []
    d_out = d_net.forward(img)[0]
    if d_out:
        d = d_out[0]
        bb_vinlpd = BBox(d.x, d.y, d.w, d.h, class_str=d.class_str, class_idx=d.class_idx)
        boxes.append(bb_vinlpd)
    parallelograms, prob = find_lp_corner(img, iwpod_tf)
    if parallelograms:
        p = parallelograms[0]
        qb_iwpod = Quadrilateral(p[0], p[1], p[2], p[3])
        boxes.append(qb_iwpod)

    return boxes
