import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from natsort import natsorted

from Data_Labeling.Dataset_Loader.DatasetLoader_WebCrawl import DatasetLoader_WebCrawl
from Data_Labeling.Graphical_Model_Generation.Graphical_Model_Generator_KOR import Graphical_Model_Generator_KOR
from Data_Labeling.find_homography_iteratively import find_total_transformation_4points
from LP_Detection.Bases import Quadrilateral
from Utils import imread_uni
from utils import crop_img_square

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_dir', type=str, default=r"E:\Dataset\01_LicensePlate\55_WebPlatemania_1944\jpg_json\all", help='source folder')
    ap.add_argument('--dst_dir', type=str, default=r"./pair_data/", help='destination folder')
    opt = ap.parse_args()

    SRC_DIR = opt.src_dir
    DST_DIR = opt.dst_dir

    # jpg, json 리스트 불러오기
    psrc = Path(SRC_DIR)
    jpg_paths = natsorted(p.absolute() for p in psrc.glob('**/*') if p.suffix.lower() in ['.jpg', '.jpeg'])
    json_paths = natsorted(p.absolute() for p in psrc.glob('**/*') if p.suffix.lower() in ['.json'])
    assert len(jpg_paths) == len(json_paths)

    pdst = Path(DST_DIR).absolute() / '01_margin_is_platewidth'
    pdst.mkdir(parents=True, exist_ok=True)

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
        margin = int(right - left)
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
        dst_path = os.path.join(pdst, '%06d.jpg' % (f + 1))
        cv2.imshow("ABM", ABM)
        cv2.waitKey(0)
        # imwrite_uni(dst_path, ABM)
