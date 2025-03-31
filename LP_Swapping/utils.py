import importlib
import os

import numpy as np


def create_model(model_file_name, in_channels, out_channels, gpu_ids):
    lib = importlib.import_module('models.%s_model' % model_file_name.lower())
    cls_name = model_file_name + 'Model'
    if cls_name in lib.__dict__:
        model = lib.__dict__[cls_name]
        return model(in_channels, out_channels, gpu_ids)


def create_dataset(dataset_name, image_folder_path):
    lib = importlib.import_module('datasets')
    cls_name = dataset_name + 'Dataset'
    if cls_name in lib.__dict__:
        dataset = lib.__dict__[cls_name]
        return dataset(image_folder_path)


def save_log(_path, s: str):
    with open(os.path.join(_path, '_log.txt'), 'a', encoding='utf-8') as f:
        if not s.endswith('\n'):
            s += '\n'
        f.write(s)


def cvt_args2str(d: dict):
    msg = '====================================arg list====================================\n'
    for k, v in d.items():
        msg += '%s: (%s) %s\n' % (k, v.__class__.__name__, v)
    msg += '===============================================================================\n'
    return msg


def crop_img_square(big_img, cx, cy, margin=128):
    """
    중심 좌표 (cx, cy)를 기준으로 좌우 margin 만큼의 정사각형 영역을 crop 합니다.
    만약 이미지 크기가 부족한 경우, zero padding을 적용하여 정사각형 사이즈를 유지합니다.

    Parameters:
        big_img (numpy.ndarray): 입력 이미지
        cx (int): 중심 x좌표
        cy (int): 중심 y좌표
        margin (int): 좌우 margin 값 (출력 크기는 margin*2 x margin*2)

    Returns:
        numpy.ndarray: 정사각형 crop 이미지 (zero padding 포함 가능)
    """
    h, w = big_img.shape[:2]
    size = margin * 2

    # Crop 영역의 좌표 계산
    x1 = cx - margin
    y1 = cy - margin
    x2 = cx + margin
    y2 = cy + margin

    # 출력 이미지를 위한 빈 배열 생성 (zero padding용)
    c = big_img.shape[-1] if big_img.ndim == 3 else 1
    crop = np.zeros((size, size, c), dtype=big_img.dtype).squeeze()

    # 실제 이미지 내에 존재하는 영역 계산
    left = max(x1, 0)
    top = max(y1, 0)
    right = min(x2, w)
    bottom = min(y2, h)

    # crop 이미지 내에서 복사할 위치 계산
    x1_crop = left - x1
    y1_crop = top - y1
    x2_crop = x1_crop + (right - left)
    y2_crop = y1_crop + (bottom - top)

    # 이미지 복사
    crop[y1_crop:y2_crop, x1_crop:x2_crop, ...] = big_img[top:bottom, left:right, ...]

    tblr = [top, bottom, left, right]

    return crop, tblr
