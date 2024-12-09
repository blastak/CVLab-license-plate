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


def crop_img_square(big_img, cx: int, cy: int, margin: int = 360):
    sq_lr = np.array([cx - margin, cx + margin])
    sq_tb = np.array([cy - margin, cy + margin])
    if sq_lr[0] < 0:
        sq_lr -= sq_lr[0]
    if big_img.shape[1] - sq_lr[1] <= 0:
        sq_lr += (big_img.shape[1] - sq_lr[1])
    if sq_tb[0] < 0:
        sq_tb -= sq_tb[0]
    if big_img.shape[0] - sq_tb[1] <= 0:
        sq_tb += (big_img.shape[0] - sq_tb[1])

    small_img = big_img[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1], ...].copy()
    return small_img
