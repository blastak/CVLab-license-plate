from abc import ABC, abstractmethod

import torch
from torchvision.utils import save_image


class BaseGANModel(ABC):
    def __init__(self):
        super().__init__()
        self.gpu_ids = []
        self.fixed_data = None
        self.device = None
        self.net_G = None
        self.net_D = None
        self.loss_G = None
        self.loss_D = None

    @abstractmethod
    def input_data(self, data):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward_G(self):
        pass

    @abstractmethod
    def backward_D(self):
        pass

    def set_requires_grad(self, net, requires_grad: bool):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def learning(self):
        self.forward()
        self.set_requires_grad(self.net_D, False)
        self.backward_G()
        self.set_requires_grad(self.net_D, True)
        self.backward_D()

    def testing(self):
        with torch.no_grad():
            self.forward()

    def eval(self):
        self.net_G.eval()
        self.net_D.eval()

    def __repr__(self):
        return 'net_G:\n%s\n\nnet_D:\n%s' % (self.net_G, self.net_D)

    def get_current_loss(self):
        return {'loss_G': float(self.loss_G), 'loss_D': float(self.loss_D)}

    def save_generated_image(self, image_path):
        if self.fixed_data is None:
            raise NotImplementedError

        with torch.no_grad():
            img = self.net_G(self.fixed_data).detach().cpu()
        save_image(img, image_path, nrow=int(self.fixed_data.shape[0] ** 0.5), normalize=True)

    def save_checkpoints(self, ckpt_path):
        if isinstance(self.net_G, torch.nn.DataParallel):
            torch.save({
                'modelD_state_dict': self.net_D.module.cpu().state_dict(),
                'modelG_state_dict': self.net_G.module.cpu().state_dict(),
                'optimizerD_state_dict': self.__dict__['optimizer_D'].state_dict(),
                'optimizerG_state_dict': self.__dict__['optimizer_G'].state_dict(),
            }, ckpt_path)
            self.net_D.cuda(self.gpu_ids[0])
            self.net_G.cuda(self.gpu_ids[0])
        else:
            torch.save({
                'modelD_state_dict': self.net_D.state_dict(),
                'modelG_state_dict': self.net_G.state_dict(),
                'optimizerD_state_dict': self.__dict__['optimizer_D'].state_dict(),
                'optimizerG_state_dict': self.__dict__['optimizer_G'].state_dict(),
            }, ckpt_path)

    def load_checkpoints(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=str(self.device))
        if isinstance(self.net_G, torch.nn.DataParallel):
            self.net_D.module.load_state_dict(state_dict['modelD_state_dict'])
            self.net_G.module.load_state_dict(state_dict['modelG_state_dict'])
            # self.__dict__['optimizer_D'].module.load_state_dict(state_dict['optimizerD_state_dict'])
            # self.__dict__['optimizer_G'].module.load_state_dict(state_dict['optimizerG_state_dict'])
        else:
            self.net_D.load_state_dict(state_dict['modelD_state_dict'])
            self.net_G.load_state_dict(state_dict['modelG_state_dict'])
            # self.__dict__['optimizer_D'].load_state_dict(state_dict['optimizerD_state_dict'])
            # self.__dict__['optimizer_G'].load_state_dict(state_dict['optimizerG_state_dict'])
