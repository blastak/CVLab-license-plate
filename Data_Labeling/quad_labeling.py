import argparse

import cv2
import numpy as np

from Data_Labeling.Graphical_Model_Generation.Graphical_Model_Generator_KOR import Graphical_Model_Generator_KOR
from Data_Labeling.labeling_utils import generate_license_plate, find_total_transformation, front_image, compare_ocrbb, detect_all
from LP_Detection.IWPOD_tf.src.keras_utils import load_model_tf
from LP_Detection.VIN_LPD.VinLPD import load_model_VinLPD
from LP_Detection.ultralytics import YOLO
from LP_Recognition.VIN_OCR.VinOCR import load_model_VinOCR
from Utils import imread_uni, save_quad

extensions = ['.jpg', '.png', '.xml', '.json']

from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default=r'D:\Dataset\LicensePlate\test\test_make_label', help='Input Image folder')
    opt = parser.parse_args()

    prefix_path = Path(opt.data)
    img_paths = [p for p in prefix_path.iterdir() if p.suffix.lower() == '.jpg']

    d_net = load_model_VinLPD('../LP_Detection/VIN_LPD/weight')  # VIN_LPD 사용 준비
    r_net = load_model_VinOCR('../LP_Recognition/VIN_OCR/weight')
    iwpod_tf = load_model_tf('../LP_Detection/IWPOD_tf/weights/iwpod_net')  # iwpod_tf 사용 준비
    yolo = YOLO(f"../../LP_Detection/ultralytics/pt/min_ALL_11l.pt")
    generator = Graphical_Model_Generator_KOR()

    for _, img_path in enumerate(img_paths):
        img = imread_uni(img_path)  # 이미지 로드
        i_h, i_w = img.shape[:2]
        file_stem = img_path.stem  # 예: 'car_plate_12_서울12가1234'
        plate_type, plate_number = file_stem.split('_')[-2], file_stem.split('_')[-1]

        boxes = detect_all(img, d_net, iwpod_tf)

        img_results = []
        dst_xy_list = []
        for _, bb_or_qb in enumerate(boxes):
            img_gened = generate_license_plate(generator, plate_type, plate_number)
            g_h, g_w = img_gened.shape[:2]

            mat_T = find_total_transformation(img_gened, generator, plate_type, img, bb_or_qb)

            for T in mat_T:
                # graphical model을 전체 이미지 좌표계로 warping
                img_gen_recon = cv2.warpPerspective(img_gened, T, (i_w, i_h))

                # 영상 합성
                fg_rgb = img_gen_recon[:, :, :3]  # RGB 채널
                alpha = img_gen_recon[:, :, 3] / 255  # alpha 채널
                img_superimposed = (alpha[:, :, np.newaxis] * fg_rgb + (1 - alpha[:, :, np.newaxis]) * img).astype(np.uint8)

                # 좌표 계산
                dst_xy = cv2.perspectiveTransform(np.float32([[[0, 0], [g_w, 0], [g_w, g_h], [0, g_h]]]), T)
                cv2.polylines(img_superimposed, [np.int32(dst_xy)], True, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)  # quadrilateral box

                img_results.append([img_superimposed, img])
                dst_xy_list.append(dst_xy)
        print(plate_number)

        min_offset = 30
        best_result = None
        for i, dst_xy in enumerate(dst_xy_list):
            img_front = front_image(dst_xy, img, plate_type, img_path, prefix_path, generator, i, save=False)
            offset = compare_ocrbb(img_front, generator, plate_type, r_net)
            print(offset)
            if offset < min_offset:
                min_offset = offset
                best_result = (dst_xy, i, min_offset)

        if best_result:  # best 저장
            print(best_result)
            best_dst_xy, best_index, min_offset = best_result
            save_quad(best_dst_xy, plate_type, plate_number, prefix_path, img_path.name, i_h, i_w)
            front_image(best_dst_xy, img, plate_type, img_path, prefix_path, generator, 'best', save=True)
