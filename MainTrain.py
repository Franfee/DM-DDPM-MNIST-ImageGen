# -*- coding: utf-8 -*-
# @Time    : 2023/8/10 11:26
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
import os
import time
import torch
from torch import nn

from utils.getDataLoader import get_dataloader

from net.DDPM import DDPM
from net.net_build import build_network
from net.net_config import configs

# 如何在PyTorch中使用自动混合精度？
# 答案就是autocast + GradScaler
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

# -------------------------------
BATCH_SIZ = 512
EPOCHS = 100
DEVICE = 'cuda:0'
# -------------------------------


def train(ddpm: DDPM, net, device, ckpt_path):
    print('batch size:', BATCH_SIZ)
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(batch_size=BATCH_SIZ, train=True)
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)

    # 在训练最开始之前实例化一个GradScaler对象
    scaler = GradScaler()

    tic = time.time()
    for e in range(EPOCHS):
        total_loss = 0

        for x, _ in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)

            # 随机时间步
            t = torch.randint(0, n_steps, (current_batch_size,)).to(device)
            # 采样高斯噪声
            eps = torch.randn_like(x).to(device)
            # 扩散过程加入噪声
            x_t = ddpm.sample_forward(x, t, eps)

            # 前向过程(model + loss)开启 autocast
            with autocast():
                eps_theta = net(x_t, t.reshape(current_batch_size, 1))
                loss = loss_fn(eps_theta, eps)

            # # 神经网络预估噪声
            # eps_theta = net(x_t, t.reshape(current_batch_size, 1))

            # loss = loss_fn(eps_theta, eps)

            # -----------------
            optimizer.zero_grad()
            # loss.backward()
            # Scales loss，这是因为半精度的数值范围有限，因此需要用它放大,否则报错
            scaler.scale(loss).backward()

            # optimizer.step()

            # scaler.step() unscale之前放大后的梯度，但是scale太多可能出现inf或NaN,故其会判断是否出现了inf/NaN
            # 如果梯度的值不是 inf 或者 NaN, 那么调用optimizer.step()来更新权重,
            # 如果检测到出现了inf或者NaN，scaler就自动跳过这次梯度更新，同时动态调整scaler的大小
            scaler.step(optimizer)

            # 查看是否要动态调整scaler的大小scaler,这个要注意不能丢
            scaler.update()
            # ----------------

            total_loss += loss.item() * current_batch_size
            # end one batch
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(net.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
        # end one epoch
    print('Done')


if __name__ == '__main__':
    os.makedirs('result', exist_ok=True)
    os.makedirs('result/models', exist_ok=True)

    # -------ddpm---------
    diffusion_steps = 1000
    ddpm_helper = DDPM(DEVICE, diffusion_steps)

    # ------build net------
    config_id = 4
    config = configs[config_id]
    scratch_net = build_network(config, diffusion_steps)

    # ------train---------
    model_path = 'result/models/model_unet_res.pth'
    train(ddpm_helper, scratch_net, device=DEVICE, ckpt_path=model_path)
