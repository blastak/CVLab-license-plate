import importlib
import os


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
    with open(os.path.join(_path, 'log.txt'), 'a', encoding='utf-8') as f:
        if not s.endswith('\n'):
            s += '\n'
        f.write(s)


def cvt_args2str(d: dict):
    msg = '====================================arg list====================================\n'
    for k, v in d.items():
        msg += '%s: (%s) %s\n' % (k, v.__class__.__name__, v)
    msg += '===============================================================================\n'
    return msg
