import torch
import torch.nn as nn


class C0nvBnLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, activation='leaky_relu', conv_type='conv', filter_size=4, stride=2, padding=1, batch_on=True, bias=False):
        super().__init__()
        if activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

        if conv_type == 'conv':
            self.conv = nn.Conv2d(in_channels, out_channels, filter_size, stride, padding, bias=bias)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, filter_size, stride, padding, bias=bias)

        self.batch_on = batch_on
        if batch_on:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.act(x)
        x = self.conv(x)
        if self.batch_on:
            x = self.bn(x)
        return x


class GeneratorUnet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        """
        GeneratorUnet
        :param in_channels: condition image의 채널수(EnhancementGAN의 경우 1, IDPreserveGAN의 경우 2~3)
        """
        super().__init__()
        inner_filters = 64
        nf = [inner_filters * 2 ** a for a in range(4)]
        self.encoder1 = nn.Conv2d(in_channels, nf[0], 4, 2, 1, bias=False)
        self.encoder2 = C0nvBnLeaky(nf[0], nf[1])
        self.encoder3 = C0nvBnLeaky(nf[1], nf[2])
        self.encoder4 = C0nvBnLeaky(nf[2], nf[3])
        self.encoder5 = C0nvBnLeaky(nf[3], nf[3])
        self.encoder6 = C0nvBnLeaky(nf[3], nf[3])
        self.encoder7 = C0nvBnLeaky(nf[3], nf[3])
        self.encoder8 = C0nvBnLeaky(nf[3], nf[3], batch_on=False)
        self.decoder8 = C0nvBnLeaky(nf[3], nf[3], 'relu', 'deconv')
        self.decoder7 = C0nvBnLeaky(nf[3] * 2, nf[3], 'relu', 'deconv')
        self.decoder6 = C0nvBnLeaky(nf[3] * 2, nf[3], 'relu', 'deconv')
        self.decoder5 = C0nvBnLeaky(nf[3] * 2, nf[3], 'relu', 'deconv')
        self.decoder4 = C0nvBnLeaky(nf[3] * 2, nf[2], 'relu', 'deconv')
        self.decoder3 = C0nvBnLeaky(nf[2] * 2, nf[1], 'relu', 'deconv')
        self.decoder2 = C0nvBnLeaky(nf[1] * 2, nf[0], 'relu', 'deconv')
        self.decoder1 = nn.Sequential(
            C0nvBnLeaky(nf[0] * 2, out_channels, 'relu', 'deconv', batch_on=False, bias=True),
            nn.Tanh()
        )
        self.drop_out = nn.Dropout(p=0.5)

    def forward(self, x):
        e1 = self.encoder1(x)  # in_channels->64, 128x128 (입력이 256일 경우 예시)
        e2 = self.encoder2(e1)  # 64->128, 64x64
        e3 = self.encoder3(e2)  # 128->256, 32x32
        e4 = self.encoder4(e3)  # 256->512, 16x16
        e5 = self.encoder5(e4)  # 512->512, 8x8
        e6 = self.encoder6(e5)  # 512->512, 4x4
        e7 = self.encoder7(e6)  # 512->512, 2x2
        x = self.encoder8(e7)  # 512->512, 1x1

        x = self.decoder8(x)  # 512->512, 2x2
        x = self.decoder7(torch.cat((x, e7), dim=1))  # (512+512)->512, 4x4
        x = self.drop_out(x)
        x = self.decoder6(torch.cat((x, e6), dim=1))  # (512+512)->512, 8x8
        x = self.drop_out(x)
        x = self.decoder5(torch.cat((x, e5), dim=1))  # (512+512)->512, 16x16
        x = self.drop_out(x)
        x = self.decoder4(torch.cat((x, e4), dim=1))  # (512+512)->256, 32x32
        x = self.decoder3(torch.cat((x, e3), dim=1))  # (256+256)->128, 64x64
        x = self.decoder2(torch.cat((x, e2), dim=1))  # (128+128)->64, 128x128
        x = self.decoder1(torch.cat((x, e1), dim=1))  # (64+64)->1, 256x256
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels, mode='patch'):
        """
        Discriminator
        :param in_channels: A,B가 concat된 총 채널수(EnhancementGAN의 경우 2, IDPreserveGAN의 경우 3~4, ThumbnailGAN은 1)
        :param mode: pix2pix용으로 쓸건지[default]. dcgan용으로 쓸건지
        """
        super().__init__()
        inner_filters = 64 if mode == 'patch' else 128
        nf = [inner_filters * 2 ** a for a in range(4)]
        self.layer1 = nn.Conv2d(in_channels, nf[0], 4, 2, 1, bias=True)
        self.layer2 = C0nvBnLeaky(nf[0], nf[1])
        self.layer3 = C0nvBnLeaky(nf[1], nf[2])
        if mode == 'patch':
            self.layer4 = C0nvBnLeaky(nf[2], nf[3], stride=1)
            self.layer5 = C0nvBnLeaky(nf[3], 1, stride=1, batch_on=False, bias=True)
        else:
            image_width = 64
            tensor_width = image_width // (2 ** 4)  # 64->32->16->8->4
            self.layer4 = C0nvBnLeaky(nf[2], nf[3])
            self.layer5 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nf[3] * tensor_width ** 2, 1, bias=False),
                nn.Sigmoid()
            )  # dense

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class GeneratorDC(nn.Module):
    def __init__(self, in_dims, out_channels):
        """
        GeneratorDC
        :param in_dims: 512-dim random vector
        :param out_channels: output image channels
        """
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_dims, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512)
        )
        self.layer2 = C0nvBnLeaky(512, 256, 'relu', 'deconv')
        self.layer3 = C0nvBnLeaky(256, 128, 'relu', 'deconv')
        self.layer4 = C0nvBnLeaky(128, 64, 'relu', 'deconv')
        self.layer5 = C0nvBnLeaky(64, out_channels, 'relu', 'deconv', batch_on=False)
        self.tanh = nn.Tanh()

    def forward(self, x):  # 512x1x1
        x = self.layer1(x)  # 1x1->4x4
        x = self.layer2(x)  # 4x4->8x8
        x = self.layer3(x)  # 8x8->16x16
        x = self.layer4(x)  # 16x16->32x32
        x = self.layer5(x)  # 32x32->outx64x64
        x = self.tanh(x)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        # nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def create_and_init(net, gpu_ids=[]):
    if len(gpu_ids) > 1:
        net = nn.DataParallel(net, gpu_ids)
    net.apply(weights_init)
    return net
