import torch
import torch.nn as nn

from LP_Swapping.models.base_GAN_model import BaseGANModel
from LP_Swapping.models.networks import GeneratorUnet, Discriminator, create_and_init


class Pix2pixModel(BaseGANModel):
    def __init__(self, in_channels: int, out_channels: int, gpu_ids=[]):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.device = torch.device('cuda:%d' % self.gpu_ids[0]) if self.gpu_ids else torch.device('cpu')
        self.net_G = create_and_init(GeneratorUnet(in_channels, out_channels), gpu_ids)
        self.net_D = create_and_init(Discriminator(in_channels + out_channels, 'patch'), gpu_ids)
        self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.lossF_GAN = nn.BCEWithLogitsLoss()
        self.lossF_L1 = nn.L1Loss()
        self.lambda1_L1 = 10

    def input_data(self, data):
        self.condi_image = data['condition_image'].to(self.device)
        self.real_image = data['real_image'].to(self.device)
        if self.fixed_data is None:
            self.fixed_data = self.condi_image.clone()

    def forward(self):
        self.fake_image = self.net_G(self.condi_image)

    def backward_G(self):
        self.optimizer_G.zero_grad()

        fake_concat = torch.cat((self.condi_image, self.fake_image), dim=1)
        pred_fake = self.net_D(fake_concat)
        self.loss_GAN = self.lossF_GAN(pred_fake, torch.tensor(1.).expand_as(pred_fake))

        fake_B_gen = self.fake_image.clone()
        real_B_gen = self.real_image.clone()
        self.loss_L1 = self.lossF_L1(fake_B_gen, real_B_gen) * self.lambda1_L1

        self.loss_G = self.loss_GAN + self.loss_L1
        self.loss_G.backward()

        self.optimizer_G.step()

    def backward_D(self):
        self.optimizer_D.zero_grad()

        real_concat = torch.cat((self.condi_image, self.real_image), dim=1)
        pred_real = self.net_D(real_concat)
        loss_GAN_real = self.lossF_GAN(pred_real, torch.tensor(1.).expand_as(pred_real))

        fake_concat = torch.cat((self.condi_image, self.fake_image), dim=1)
        pred_fake = self.net_D(fake_concat.detach())
        loss_GAN_fake = self.lossF_GAN(pred_fake, torch.tensor(0.).expand_as(pred_fake))

        self.loss_D = (loss_GAN_real + loss_GAN_fake) * 0.5
        self.loss_D.backward()

        self.optimizer_D.step()
