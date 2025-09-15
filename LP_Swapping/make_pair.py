import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
from natsort import natsorted

from Data_Labeling.Dataset_Loader.DatasetLoader_WebCrawl import DatasetLoader_WebCrawl
from Data_Labeling.Graphical_Model_Generation.Graphical_Model_Generator_KOR import Graphical_Model_Generator_KOR
from Data_Labeling.find_homography_iteratively import find_total_transformation_4points, calculate_text_area_coordinates
from LP_Detection.Bases import Quadrilateral
from Utils import imread_uni, imwrite_uni, encrypt_number
from utils import crop_img_square

# pw = '8470'

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument('--src_dir', type=str, default=r"E:\Dataset\01_LicensePlate\55_WebPlatemania_1944_完\55_WebPlatemania_jpg_json\good_all", help='source folder')
    ap.add_argument('--src_dir', type=str, default=r"E:\Dataset\01_LicensePlate\05_Masan_1056_完\good_all", help='source folder')
    # ap.add_argument('--src_dir', type=str, default=r"E:\Dataset\01_LicensePlate\56_WebEV_\56_WebEV_jpg_json\GoodMatches_P1-2", help='source folder')
    # ap.add_argument('--src_dir', type=str, default=r"E:\Dataset\01_LicensePlate\99_Techwin_1F_241\99_Techwin_1F_jpg_json\good_all", help='source folder')
    # ap.add_argument('--src_dir', type=str, default=r"E:\Dataset\01_LicensePlate\99_Techwin_B1_in_\99_Techwin_B1_in_jpg_json\good_all", help='source folder')
    # ap.add_argument('--src_dir', type=str, default=r"E:\Dataset\01_LicensePlate\99_Techwin_B1_out_\99_Techwin_B1_out_jpg_json\good_all", help='source folder')
    ap.add_argument('--dst_dir', type=str, default=r"./pair_data/03color_same-id_P4", help='destination folder')
    # ap.add_argument('--dst_dir', type=str, default=r"./pair_data/03color_de-id_pw;"+pw, help='destination folder')
    opt = ap.parse_args()

    SRC_DIR = opt.src_dir
    DST_DIR = opt.dst_dir

    # jpg, json 리스트 불러오기
    psrc = Path(SRC_DIR)
    jpg_paths = natsorted(p.absolute() for p in psrc.glob('**/*') if p.suffix.lower() in ['.jpg', '.jpeg'])
    json_paths = natsorted(p.absolute() for p in psrc.glob('**/*') if p.suffix.lower() in ['.json'])
    assert len(jpg_paths) == len(json_paths)

    # # 공통 인덱스를 샘플링
    # indices = random.sample(range(len(jpg_paths)), 1000)  # 탄천같이 큰 데이터에 대해서만 하기로 함
    # # 두 리스트에서 같은 인덱스 순서로 추출
    # jpg_paths = [jpg_paths[i] for i in indices]
    # json_paths = [json_paths[i] for i in indices]

    pdst = {'train': Path(DST_DIR).absolute() / 'train', 'test': Path(DST_DIR).absolute() / 'test'}
    pdst['train'].mkdir(parents=True, exist_ok=True)
    pdst['test'].mkdir(parents=True, exist_ok=True)

    generator = Graphical_Model_Generator_KOR()
    loader = DatasetLoader_WebCrawl(SRC_DIR)
    # for f, (jpg_path, json_path) in enumerate(zip(jpg_paths, json_paths)):
    f = 0
    for jpg_path, json_path in zip(jpg_paths, json_paths):
        plate_type, plate_number, xy1, xy2, xy3, xy4, left, top, right, bottom = loader.parse_json(json_path)
        if plate_type != 'P4':
            continue
        frame = imread_uni(jpg_path)

        # # non encryption
        img_gen2x = generator.make_LP(plate_number, plate_type)
        # encryption
        # enc_number = encrypt_number(plate_type, plate_number, pw)#, MODE == 'decrypted')
        # img_gen2x = generator.make_LP(enc_number, plate_type)

        img_gen1x = cv2.resize(img_gen2x, None, fx=0.5, fy=0.5)

        # superimposing
        qb = Quadrilateral(xy1, xy2, xy3, xy4)
        mat_T = find_total_transformation_4points(img_gen1x, frame, qb)
        img_gen1x_recon = cv2.warpPerspective(img_gen1x, mat_T, frame.shape[1::-1])

        cx = int((left + right) / 2)
        cy = int((top + bottom) / 2)

        ###### 01_margin_is_platewidth (즉, 정방형 크기는 번호판 가로의 2배이다)
        # margin = int(right - left)

        ###### 02_fix_margin_240 (즉, 정방형 크기는 480)
        # margin = 240
        # mask_white = img_gen1x_recon[:, :, 3]

        ###### 03_fix_margin_240_txtarea
        margin = 240
        mask_text_area = calculate_text_area_coordinates(generator, plate_type)
        mask_white = cv2.warpPerspective(mask_text_area, mat_T, frame.shape[1::-1])

        frame_roi, _ = crop_img_square(frame, cx, cy, margin)
        mask_white_roi, _ = crop_img_square(mask_white, cx, cy, margin)
        img_gen_roi, _ = crop_img_square(img_gen1x_recon[:, :, :3], cx, cy, margin)
        img1 = cv2.bitwise_and(frame_roi, frame_roi, mask=cv2.bitwise_not(mask_white_roi))
        img2 = cv2.bitwise_and(img_gen_roi, img_gen_roi, mask=mask_white_roi)
        A = img1 + img2
        B = frame_roi
        M = cv2.cvtColor(mask_white_roi, cv2.COLOR_GRAY2BGR)

        ABM = np.hstack((A, B, M))

        if (f + 1) % 10 == 0:
            dst_path = os.path.join(pdst['test'], '%s_.png' % jpg_path.stem)
        else:
            dst_path = os.path.join(pdst['train'], '%s_.png' % jpg_path.stem)
        print(f + 1, '/', len(jpg_paths), '\t', dst_path)
        # cv2.imshow("ABM", ABM)
        # cv2.waitKey(0)
        f+=1
        imwrite_uni(dst_path, ABM)
