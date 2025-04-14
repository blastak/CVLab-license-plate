import argparse
import os
from pathlib import Path

import cv2
from natsort import natsorted

from Data_Labeling.Dataset_Loader.DatasetLoader_WebCrawl import DatasetLoader_WebCrawl
from Data_Labeling.Graphical_Model_Generation.Graphical_Model_Generator_KOR import Graphical_Model_Generator_KOR
from Data_Labeling.find_homography_iteratively import find_total_transformation_4points
from LP_Detection.Bases import Quadrilateral
from LP_Swapping.swap import Swapper
from Utils import imread_uni, imwrite_uni, encrypt_number
from utils import crop_img_square


def pixelate_image(img, pixel_size=10):
    # 원본 크기 저장
    height, width = img.shape[:2]

    # 작은 크기로 축소 후 다시 원래 크기로 확대
    small = cv2.resize(img, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixelated


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_dir', type=str, default=r"E:\Dataset\01_LicensePlate\55_WebPlatemania_1944\55_WebPlatemania_jpg_json\good_all", help='source folder')
    ap.add_argument('--dst_dir', type=str, default=r"E:\Dataset\01_LicensePlate\55_WebPlatemania_1944\55_WebPlatemania_jpg_json\degradation", help='destination folder')
    opt = ap.parse_args()

    SRC_DIR = opt.src_dir
    DST_DIR = opt.dst_dir

    # jpg, json 리스트 불러오기
    psrc = Path(SRC_DIR)
    jpg_paths = natsorted(p.absolute() for p in psrc.glob('**/*') if p.suffix.lower() in ['.jpg', '.jpeg'])
    json_paths = natsorted(p.absolute() for p in psrc.glob('**/*') if p.suffix.lower() in ['.json'])
    assert len(jpg_paths) == len(json_paths)

    swapper = Swapper('../LP_Swapping/checkpoints/Masked_Pix2pix_CondRealMask_try005_server/ckpt_best_loss_G.pth')
    pdst = Path(DST_DIR).absolute() / 'encrypted'
    pdst.mkdir(parents=True, exist_ok=True)

    generator = Graphical_Model_Generator_KOR()
    loader = DatasetLoader_WebCrawl(SRC_DIR)
    for f, (jpg_path, json_path) in enumerate(zip(jpg_paths, json_paths)):
        frame = imread_uni(jpg_path)
        frame2 = frame.copy()

        plate_type, plate_number, xy1, xy2, xy3, xy4, left, top, right, bottom = loader.parse_json(json_path)

        enc_number = encrypt_number(plate_type, plate_number, 'cvlab', False)

        img_gen2x = generator.make_LP(enc_number, plate_type)
        img_gen1x = cv2.resize(img_gen2x, None, fx=0.5, fy=0.5)

        # superimposing
        qb = Quadrilateral(xy1, xy2, xy3, xy4)
        mat_T = find_total_transformation_4points(img_gen1x, generator, plate_type, frame, qb)
        img_gen1x_recon = cv2.warpPerspective(img_gen1x, mat_T, frame.shape[1::-1])
        mask_white = img_gen1x_recon[:, :, 3]

        cx = int((left + right) / 2)
        cy = int((top + bottom) / 2)
        margin = int(right - left)
        frame_roi, tblr = crop_img_square(frame, cx, cy, margin)
        mask_white_roi, _ = crop_img_square(mask_white, cx, cy, margin)
        img_gen_roi, _ = crop_img_square(img_gen1x_recon[:, :, :3], cx, cy, margin)
        img1 = cv2.bitwise_and(frame_roi, frame_roi, mask=cv2.bitwise_not(mask_white_roi))
        img2 = cv2.bitwise_and(img_gen_roi, img_gen_roi, mask=mask_white_roi)

        A = img1 + img2
        B = frame_roi
        M = cv2.cvtColor(mask_white_roi, cv2.COLOR_GRAY2BGR)

        inputs = swapper.make_tensor(A, B, M)

        img_swapped = swapper.swap(inputs)
        img_swapped_unshrink = cv2.resize(img_swapped, (tblr[3] - tblr[2], tblr[1] - tblr[0]))

        frame2[tblr[0]:tblr[1], tblr[2]:tblr[3], ...] = img_swapped_unshrink.copy()

        print(f + 1, '/', len(jpg_paths), '\t', jpg_path.name)
        # cv2.imshow('frame2', frame2)
        # cv2.waitKey(0)
        dst_path = os.path.join(pdst, '%s_.jpg' % jpg_path.stem)
        imwrite_uni(dst_path, frame2)
