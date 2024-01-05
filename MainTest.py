# -*- coding: utf-8 -*-
# @Time    : 2023/8/10 11:26
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
import os

import torch

from net.UNet import UNet
from net.DDPM import DDPM
from net.base import get_img_shape
from torchvision.utils import make_grid, save_image

DEVICE = 'cuda:0'
#DEVICE = 'cpu'

def sample_imgs(ddpm, net, output_path, n_sample=81, device='cuda:0', simple_var=True):
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        shape = (n_sample, *get_img_shape())
        print("sampling shape:", shape)

        imgs = ddpm.sample_backward(shape, net, device=device, simple_var=simple_var).detach().cpu()
        img_grid = make_grid(imgs,nrow=9, normalize=True)
        save_image(img_grid, output_path, normalize=False)
        print("sample done.")


if __name__ == '__main__':
    os.makedirs('result/images', exist_ok=True)

    # -------ddpm---------
    n_steps = 1000
    ddpm_h = DDPM(DEVICE, n_steps)

    # ------build net------
    cloneNet = UNet(n_steps=n_steps, channels=[10, 20, 40, 80], pe_dim=128, residual=True)

    model_path = 'result/models/model_unet_res.pth'
    cloneNet.load_state_dict(torch.load(model_path, map_location='cpu'))

    # sample
    sample_imgs(ddpm_h, cloneNet, 'result/images/diffusion.png', device=DEVICE)
