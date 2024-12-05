import torch

from LP_Swapping.models.pix2pix_model import Pix2pixModel


class Masked_Pix2pixModel(Pix2pixModel):
    def __init__(self, in_channels: int, out_channels: int, gpu_ids=[]):
        super().__init__(in_channels, out_channels, gpu_ids)
        self.lambda1_L1 = 100
        self.lambda2_L1 = 10

    def backward_G(self):
        self.optimizer_G.zero_grad()

        fake_concat = torch.cat((self.condi_image, self.fake_image), dim=1)
        pred_fake = self.net_D(fake_concat)
        self.loss_GAN = self.lossF_GAN(pred_fake, torch.tensor(1.).expand_as(pred_fake))

        M = self.condi_image.clone()[:, -1, :, :]
        M = M.unsqueeze(dim=1)
        M = (M > 0).float()  # thresholding
        fake_B_gen = self.fake_image.clone()
        real_B_gen = self.real_image.clone()
        cond_generated = self.lossF_L1(M * fake_B_gen, M * real_B_gen)
        M = 1 - M
        cond_inherited = self.lossF_L1(M * fake_B_gen, M * real_B_gen)
        self.loss_L1 = cond_generated * self.lambda1_L1 + cond_inherited * self.lambda2_L1

        self.loss_G = self.loss_GAN + self.loss_L1
        self.loss_G.backward()

        self.optimizer_G.step()
