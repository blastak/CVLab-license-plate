import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import make_grid

from LP_Swapping.models.masked_pix2pix_model import Masked_Pix2pixModel


class Swapper:
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

    def __init__(self, ckpt_path):
        ########## torch environment settings
        gpu_ids = [0]
        self.device = torch.device('cuda:%d' % gpu_ids[0] if (torch.cuda.is_available() and len(gpu_ids) > 0) else 'cpu')
        torch.set_default_device(self.device)  # working on torch>2.0.0
        if torch.cuda.is_available() and len(gpu_ids) > 1:
            torch.multiprocessing.set_start_method('spawn')
        ########## model settings
        self.model = Masked_Pix2pixModel(4, 3, gpu_ids)
        self.model.load_checkpoints(ckpt_path)
        self.model.eval()

    def make_tensor(self, A, B, M):
        A_ = Image.fromarray(A)
        B_ = Image.fromarray(B)
        M_ = Image.fromarray(M)
        t_cond = self.tf_img(A_)
        t_real = self.tf_img(B_)
        t_mask = self.tf_mask(M_)
        t_cond = torch.cat((t_cond, t_mask), dim=0)

        t_cond = torch.unsqueeze(t_cond, dim=0)
        t_real = torch.unsqueeze(t_real, dim=0)
        sample = {'condition_image': t_cond, 'real_image': t_real}
        return sample

    def swap(self, inputs):
        self.model.input_data(inputs)
        self.model.testing()
        detached = self.model.fake_image.detach().cpu()
        bs = 1
        montage = make_grid(detached, nrow=int(bs ** 0.5), normalize=True).permute(1, 2, 0).numpy()
        montage = cv2.normalize(montage, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
        return montage
