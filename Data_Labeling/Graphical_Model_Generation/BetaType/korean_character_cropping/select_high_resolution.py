import os
import shutil

from Utils import imread_uni
from delete_hash import remove_duplicate_images

if __name__ == "__main__":
    prefix_path = r"D:\Dataset\LicensePlate\for_p5p6\data2\P5"
    move_path = r"D:\Dataset\LicensePlate\for_p5p6\extract\HR"
    img_paths = [a for a in os.listdir(prefix_path) if a.endswith('.jpg')]

    for _, img_path in enumerate(img_paths):
        img = imread_uni(os.path.join(prefix_path, img_path))  # 이미지 로드
        i_h, i_w = img.shape[:2]
        if i_h * i_w > 1000000:
            shutil.copy(os.path.join(prefix_path, img_path), os.path.join(move_path, img_path))

    remove_duplicate_images(move_path)
