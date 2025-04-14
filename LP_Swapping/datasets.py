from pathlib import Path

import albumentations as A
import cv2
import numpy as np
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
        self.image_path_list = natsorted(p.absolute() for p in Path(image_path_list).glob('**/*') if p.suffix.lower() in IMG_EXTENSIONS)


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
        self.image_path_list = natsorted(p.absolute() for p in Path(image_path_list).glob('**/*') if p.suffix.lower() in IMG_EXTENSIONS)

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
    def __init__(self, image_path_list, train=True, image_width=256):
        self.image_path_list = natsorted(
            p.absolute() for p in Path(image_path_list).glob('**/*')
            if p.suffix.lower() in IMG_EXTENSIONS
        )
        self.image_width = image_width
        self.train = train

        # augmentation transforms (shared for all 3 images)
        if train:
            self.transform = A.Compose([
                A.RandomResizedCrop(image_width, image_width, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
                A.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), p=0.5),
                A.Resize(image_width, image_width),  # safety resize
            ], additional_targets={
                'real': 'image',
                'mask': 'image'
            })
        else:
            self.transform = A.Compose([
                A.Resize(image_width, image_width)
            ], additional_targets={
                'real': 'image',
                'mask': 'image'
            })

        # Tensor conversion (PIL/np to torch.Tensor) & normalization
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        self.to_tensor_gray = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        # Load and split image into 3 parts
        img = Image.open(self.image_path_list[index]).convert('RGB')
        img_np = np.array(img)

        h, w, _ = img_np.shape
        w_unit = w // 3
        cond_np = img_np[:, :w_unit, :]
        real_np = img_np[:, w_unit:2 * w_unit, :]
        mask_np = img_np[:, 2 * w_unit:, :]

        # Apply the same augmentation to all three images
        augmented = self.transform(cond=cond_np, real=real_np, mask=mask_np)
        cond_aug = augmented['cond']
        real_aug = augmented['real']
        mask_aug = augmented['mask']

        # Convert to torch tensors
        cond_tensor = self.to_tensor(cond_aug)
        real_tensor = self.to_tensor(real_aug)
        mask_tensor = self.to_tensor_gray(cv2.cvtColor(mask_aug, cv2.COLOR_RGB2GRAY))

        # Combine condition image (RGB + mask channel)
        condition_image = torch.cat([cond_tensor, mask_tensor], dim=0)

        return {
            'condition_image': condition_image,
            'real_image': real_tensor
        }
