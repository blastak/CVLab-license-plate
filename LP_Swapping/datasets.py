from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from natsort import natsorted

IMG_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.bmp',
    '.ppm', '.tif', '.tiff',
]


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, image_path_list):
        self.image_path_list = natsorted(p.absolute() for p in Path(image_path_list).glob('**/*') if p.suffix in IMG_EXTENSIONS)


class CondRealDataset(torch.utils.data.Dataset):
    image_width = 256
    tf_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_width, image_width), antialias=True),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    def __init__(self, image_path_list):
        # self.image_path_list = natsorted(p.absolute() for p in Path(image_path_list).glob('**/*.jpg'))
        # self.image_path_list = natsorted(p.absolute() for p in Path(image_path_list).glob('**/*') if p.suffix == '.jpg')
        self.image_path_list = natsorted(p.absolute() for p in Path(image_path_list).glob('**/*') if p.suffix in IMG_EXTENSIONS)

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        img = Image.open(self.image_path_list[index])

        w, h = img.size
        w2 = int(w / 2)
        pair_left = img.crop((0, 0, w2, h))
        pair_right = img.crop((w2, 0, w, h))

        t_cond = self.tf_img(pair_left)
        t_real = self.tf_img(pair_right)

        sample = {'condition_image': t_cond, 'real_image': t_real}
        return sample


class CondRealMaskDataset(torch.utils.data.Dataset):
    image_width = 256
    tf_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_width, image_width), antialias=True),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    tf_mask = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Resize((image_width, image_width), antialias=True),
        transforms.Normalize([0.5], [0.5])
    ])

    def __init__(self, image_path_list):
        # self.image_path_list = natsorted(p.absolute() for p in Path(image_path_list).glob('**/*.jpg'))
        # self.image_path_list = natsorted(p.absolute() for p in Path(image_path_list).glob('**/*') if p.suffix == '.jpg')
        self.image_path_list = natsorted(p.absolute() for p in Path(image_path_list).glob('**/*') if p.suffix in IMG_EXTENSIONS)

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        img = Image.open(self.image_path_list[index])

        w, h = img.size
        w2 = int(w / 3)
        pair_left = img.crop((0, 0, w2, h))
        pair_right = img.crop((w2, 0, w2 * 2, h))
        mask = img.crop((w2 * 2, 0, w, h))

        t_cond = self.tf_img(pair_left)
        t_real = self.tf_img(pair_right)
        t_mask = self.tf_mask(mask)

        t_cond = torch.cat((t_cond, t_mask), dim=0)

        sample = {'condition_image': t_cond, 'real_image': t_real}
        return sample
