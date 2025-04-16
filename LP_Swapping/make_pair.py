import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
from natsort import natsorted

from Data_Labeling.Dataset_Loader.DatasetLoader_WebCrawl import DatasetLoader_WebCrawl
from Data_Labeling.Graphical_Model_Generation.Graphical_Model_Generator_KOR import Graphical_Model_Generator_KOR
from Data_Labeling.find_homography_iteratively import find_total_transformation_4points
from LP_Detection.Bases import Quadrilateral
from Utils import imread_uni, imwrite_uni
from utils import crop_img_square

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_dir', type=str, default=r"E:\Dataset\01_LicensePlate\04_Chungnam_192_完\good_all", help='source folder')
    ap.add_argument('--dst_dir', type=str, default=r"./pair_data/02_fix_margin_240", help='destination folder')
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
    for f, (jpg_path, json_path) in enumerate(zip(jpg_paths, json_paths)):
        frame = imread_uni(jpg_path)
        plate_type, plate_number, xy1, xy2, xy3, xy4, left, top, right, bottom = loader.parse_json(json_path)
        img_gen2x = generator.make_LP(plate_number, plate_type)
        img_gen1x = cv2.resize(img_gen2x, None, fx=0.5, fy=0.5)

        # superimposing
        qb = Quadrilateral(xy1, xy2, xy3, xy4)
        mat_T = find_total_transformation_4points(img_gen1x, generator, plate_type, frame, qb)
        img_gen1x_recon = cv2.warpPerspective(img_gen1x, mat_T, frame.shape[1::-1])
        mask_white = img_gen1x_recon[:, :, 3]

        cx = int((left + right) / 2)
        cy = int((top + bottom) / 2)

        ###### 01_margin_is_platewidth (즉, 정방형 크기는 번호판 가로의 2배이다)
        # margin = int(right - left)

        ###### 02_fix_margin_240 (즉, 정방형 크기는 480)
        margin = 240

        frame_roi, _ = crop_img_square(frame, cx, cy, margin)
        mask_white_roi, _ = crop_img_square(mask_white, cx, cy, margin)
        img_gen_roi, _ = crop_img_square(img_gen1x_recon[:, :, :3], cx, cy, margin)
        img1 = cv2.bitwise_and(frame_roi, frame_roi, mask=cv2.bitwise_not(mask_white_roi))
        img2 = cv2.bitwise_and(img_gen_roi, img_gen_roi, mask=mask_white_roi)
        A = img1 + img2
        B = frame_roi
        M = cv2.cvtColor(mask_white_roi, cv2.COLOR_GRAY2BGR)

        ABM = np.hstack((A, B, M))

        print(f + 1, '/', len(jpg_paths), '\t', jpg_path.name)
        if (f + 1) % 10 == 0:
            dst_path = os.path.join(pdst['test'], '%s_.jpg' % jpg_path.stem)
        else:
            dst_path = os.path.join(pdst['train'], '%s_.jpg' % jpg_path.stem)
        # cv2.imshow("ABM", ABM)
        # cv2.waitKey(0)
        imwrite_uni(dst_path, ABM)
