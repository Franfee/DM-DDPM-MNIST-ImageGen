# -*- coding: utf-8 -*-
# @Time    : 2023/8/10 11:26
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
import os

import cv2
import numpy as np
import torch
from einops import einops


from net.DDPM import DDPM
from net.base import get_img_shape
from net.net_build import build_network
from net.net_config import configs


DEVICE = 'cuda:0'


def sample_imgs(ddpm, net, output_path, n_sample=81, device='cuda:0', simple_var=True):
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        shape = (n_sample, *get_img_shape())
        print("sampling shape:", shape)
        imgs = ddpm.sample_backward(shape, net, device=device, simple_var=simple_var).detach().cpu()
        imgs = (imgs + 1) / 2 * 255
        imgs = imgs.clamp(0, 255)
        imgs = einops.rearrange(imgs, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=int(n_sample**0.5))

        imgs = imgs.numpy().astype(np.uint8)

        cv2.imwrite(output_path, imgs)
        print("sample done.")


if __name__ == '__main__':
    os.makedirs('result/images', exist_ok=True)

    # -------ddpm---------
    n_steps = 1000
    ddpm_h = DDPM(DEVICE, n_steps)

    # ------build net------
    config_id = 4
    config = configs[config_id]
    cloneNet = build_network(config, n_steps)

    model_path = 'result/models/model_unet_res.pth'
    cloneNet.load_state_dict(torch.load(model_path))

    # sample
    sample_imgs(ddpm_h, cloneNet, 'result/images/diffusion.jpg', device=DEVICE)
