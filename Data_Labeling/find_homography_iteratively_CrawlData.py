from Utils import imwrite_uni
from find_homography_iteratively import *


def save_quad(dst_xy, plate_type, plate_number, path, imagePath, imageHeight, imageWidth):
    shapes = []
    quad_xy = dst_xy.tolist()
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
    prefix_path = r"D:\Dataset\LicensePlate\Dataset_reorganization\test"
    img_paths = [a for a in os.listdir(prefix_path) if a.endswith('.jpg')]

    d_net = load_model_VinLPD('../LP_Detection/VIN_LPD/weight')  # VIN_LPD 사용 준비
    r_net = load_model_VinOCR('../LP_Recognition/VIN_OCR/weight')
    iwpod_tf = load_model_tf('../LP_Detection/IWPOD_tf/weights/iwpod_net')  # iwpod_tf 사용 준비

    for _, img_path in enumerate(img_paths):
        img = imread_uni(os.path.join(prefix_path, img_path))  # 이미지 로드
        i_h, i_w = img.shape[:2]
        plate_type = img_path.split('_')[1]
        plate_number = img_path.split('_')[2][:-4]
        boxes = []

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

        # VIN_LPD와 iwpod_tf corner 비교
        IOU = 0
        if parallelograms:
            IOU = cal_IOU(bb_vinlpd, parallelograms[0])
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
        if not d_out and not parallelograms:
            continue
        # IWPOD 존재 하지 않으면 VIN_LPD 사용
        if IOU > 0.2:
            i = 1
        else:
            i = 0
        dst_xy = dst_xy_list[i]
        # cv2.imshow("img", img_results[i][0])
        # cv2.waitKey(1)
        save_quad(dst_xy, plate_type, plate_number, prefix_path, img_path, img.shape[0], img.shape[1])

        # type별 폴더 이동
        move_path = os.path.join(prefix_path, plate_type)
        if not os.path.exists(move_path):
            os.makedirs(move_path)
        filename = os.path.splitext(img_path)[0]
        for ext in extensions:
            if os.path.exists(os.path.join(prefix_path, filename + ext)):
                os.rename(os.path.join(prefix_path, filename + ext), os.path.join(move_path, filename + ext))

        # frontalization 저장
        save_path = os.path.join(prefix_path, 'front_' + plate_type)
        dst_xy = Quadrilateral(dst_xy[0][0], dst_xy[0][1], dst_xy[0][2], dst_xy[0][3])
        img_front, mat_A = frontalization(img, dst_xy, generator.plate_wh[0], generator.plate_wh[1], 4)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        imwrite_uni(os.path.join(save_path, 'front_' + img_path), img_front)
