import argparse

from Utils import imwrite_uni, iou_4corner
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='', help='Input Image folder')
    opt = parser.parse_args()

    prefix_path = opt.data
    img_paths = [a for a in os.listdir(prefix_path) if a.endswith('.jpg')]

    d_net = load_model_VinLPD('../LP_Detection/VIN_LPD/weight')  # VIN_LPD 사용 준비
    r_net = load_model_VinOCR('../LP_Recognition/VIN_OCR/weight')
    iwpod_tf = load_model_tf('../LP_Detection/IWPOD_tf/weights/iwpod_net')  # iwpod_tf 사용 준비

    generator = Graphical_Model_Generator_KOR()
    for _, img_path in enumerate(img_paths):
        img = imread_uni(os.path.join(prefix_path, img_path))  # 이미지 로드
        i_h, i_w = img.shape[:2]
        plate_type = img_path.split('_')[1]
        plate_number = img_path.split('_')[2][:-4]
        boxes = []

        # VIN_LPD로 검출
        d_out = d_net.forward(img)[0]
        if d_out:
            d = d_out[0]
            bb_vinlpd = BBox(d.x, d.y, d.w, d.h, class_str=d.class_str, class_idx=d.class_idx)
            boxes.append(bb_vinlpd)

        # iwpod_tf로 검출
        parallelograms = find_lp_corner(img, iwpod_tf)
        if parallelograms:
            p = parallelograms[0]
            qb_iwpod = Quadrilateral(p[0], p[1], p[2], p[3])  # ex) p[0] : (342.353, 454.223)
            boxes.append(qb_iwpod)

        # IWPOD 존재 하지 않으면 VIN_LPD 사용
        if not d_out and not parallelograms:
            continue
        iou = 0
        if d_out and parallelograms:
            i = 1
            iou = iou_4corner(d, p)
            print(iou)
        else:
            i = 0

        img_results = []
        dst_xy_list = []
        for _, bb_or_qb in enumerate(boxes):
            img_gened = generate_license_plate(generator, plate_type, plate_number)
            g_h, g_w = img_gened.shape[:2]
            mask_text_area = calculate_text_area_coordinates(generator, plate_type)
            # mask_text_area = generator.get_text_area((g_h, g_w), plate_type)  # example
            img_front, mat_A = frontalization(img, bb_or_qb, g_w, g_h)
            pt1, pt2 = extract_N_track_features(img_gened, mask_text_area, img_front, plate_type)
            mat_H = find_homography_with_minimum_error(img_gened, mask_text_area, img_front, pt1, pt2)
            mat_T = calculate_total_transformation(mat_A, mat_H)

            # graphical model을 전체 이미지 좌표계로 warping
            img_gen_recon = cv2.warpPerspective(img_gened, mat_T, (i_w, i_h))

            # 영상 합성
            fg_rgb = img_gen_recon[:, :, :3]  # RGB 채널
            alpha = img_gen_recon[:, :, 3] / 255  # alpha 채널
            img_superimposed = (alpha[:, :, np.newaxis] * fg_rgb + (1 - alpha[:, :, np.newaxis]) * img).astype(np.uint8)

            # 좌표 계산
            dst_xy = cv2.perspectiveTransform(np.float32([[[0, 0], [g_w, 0], [g_w, g_h], [0, g_h]]]), mat_T)
            cv2.polylines(img_superimposed, [np.int32(dst_xy)], True, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)  # quadrilateral box

            img_results.append([img_superimposed, img])
            dst_xy_list.append(dst_xy)
        print(plate_number)

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
        img_front, mat_A = frontalization(img, dst_xy, *generator.plate_wh[plate_type], 4)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        imwrite_uni(os.path.join(save_path, 'front_' + img_path), img_front)
