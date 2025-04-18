import argparse
import csv
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


def save_list_to_csv(filepath, L):
    # usage :
    # indices = random.sample(range(len(jpg_paths)), 2000)
    # save_list_to_csv(r"E:\Dataset\01_LicensePlate\01_Tancheon_28206_完\20250416_2000개.csv", indices)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(L)  # 한 줄에 저장


def load_csv_to_list(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            return row  # 첫 줄만 읽어서 반환 (1차원 리스트)


def pixelate_image(img, pixel_size=15):
    # 원본 크기 저장
    height, width = img.shape[:2]

    # 작은 크기로 축소 후 다시 원래 크기로 확대
    small = cv2.resize(img, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixelated


def blur_image(img, sigma=19):
    return cv2.GaussianBlur(img, (0, 0), sigma)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_dir', type=str, default=r"E:\Dataset\01_LicensePlate\01_Tancheon_28206_完\good_all", help='source folder')
    ap.add_argument('--dst_dir', type=str, default=r"E:\Dataset\01_LicensePlate\01_Tancheon_28206_完\degradation", help='destination folder')
    ap.add_argument('--mode', choices=['original', 'encrypted', 'decrypted', 'gaussian', 'pixelated', 'black10', 'black15', 'black20'],
                    default='gaussian', help='degradation mode')
    opt = ap.parse_args()

    SRC_DIR = opt.src_dir
    DST_DIR = opt.dst_dir
    MODE = opt.mode

    # jpg, json 리스트 불러오기
    psrc = Path(SRC_DIR)
    jpg_paths = natsorted(p.absolute() for p in psrc.glob('**/*') if p.suffix.lower() in ['.jpg', '.jpeg'])
    json_paths = natsorted(p.absolute() for p in psrc.glob('**/*') if p.suffix.lower() in ['.json'])
    assert len(jpg_paths) == len(json_paths)

    # 공통 인덱스를 샘플링한 것을 불러오기
    indices = load_csv_to_list(r"E:\Dataset\01_LicensePlate\01_Tancheon_28206_完\20250416_2000개.csv")

    # 두 리스트에서 같은 인덱스 순서로 추출
    jpg_paths = [jpg_paths[int(i)] for i in indices]
    json_paths = [json_paths[int(i)] for i in indices]

    swapper = None
    generator = None
    if MODE == 'encrypted' or MODE == 'decrypted':
        # swapper = Swapper('../LP_Swapping/checkpoints/Masked_Pix2pix_CondRealMask_try005_server/ckpt_best_loss_G.pth')
        swapper = Swapper('../LP_Swapping/checkpoints/Masked_Pix2pix_CondRealMask_try007_server/ckpt_best_loss_G.pth')
        generator = Graphical_Model_Generator_KOR()

    pdst = Path(DST_DIR).absolute() / MODE
    pdst.mkdir(parents=True, exist_ok=True)

    loader = DatasetLoader_WebCrawl(SRC_DIR)
    for f, (jpg_path, json_path) in enumerate(zip(jpg_paths, json_paths)):
        frame = imread_uni(jpg_path)
        frame2 = frame.copy()

        plate_type, plate_number, xy1, xy2, xy3, xy4, left, top, right, bottom = loader.parse_json(json_path)

        pw = right - left
        ph = bottom - top

        if MODE == 'gaussian' or MODE == 'pixelated':
            left = int(max(left - pw / 8, 0))
            top = int(max(top - ph / 4, 0))
            right = int(min(right + pw / 8, frame.shape[1]))
            bottom = int(min(bottom + ph / 4, frame.shape[0]))
            frame2[top:bottom, left:right] = blur_image(frame2[top:bottom, left:right]) if MODE == 'gaussian' else pixelate_image(frame2[top:bottom, left:right])
        elif 'black' in MODE:
            if '10' in MODE:
                left = int(left)
                top = int(top)
                right = int(right)
                bottom = int(bottom)
            elif '15' in MODE:
                left = int(max(left - pw / 4, 0))
                top = int(max(top - ph / 4, 0))
                right = int(min(right + pw / 4, frame.shape[1]))
                bottom = int(min(bottom + ph / 4, frame.shape[0]))
            else:
                left = int(max(left - pw / 2, 0))
                top = int(max(top - ph / 2, 0))
                right = int(min(right + pw / 2, frame.shape[1]))
                bottom = int(min(bottom + ph / 2, frame.shape[0]))
            cv2.rectangle(frame2, (left, top), (right, bottom), (0, 0, 0), cv2.FILLED)

        elif MODE == 'original':
            pass
        elif MODE == 'encrypted' or MODE == 'decrypted':

            enc_number = encrypt_number(plate_type, plate_number, 'cvlab', MODE == 'decrypted')

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
