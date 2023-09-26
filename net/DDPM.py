# -*- coding: utf-8 -*-
# @Time    : 2023/8/10 11:26
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
import torch


class DDPM:
    """
    DDPM本身不是一个神经网络，它只是描述了前向过程和后向过程的一些计算。只有涉及可学习参数的神经网络类才应该继承 torch.nn.Module

    注意，为了方便实现，我们让t的取值从0开始，要比论文t里的少1。

    """
    def __init__(self, device, n_steps: int, min_beta: float = 0.0001, max_beta: float = 0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        alpha_prev = torch.empty_like(alpha_bars)
        alpha_prev[1:] = alpha_bars[0:n_steps - 1]
        alpha_prev[0] = 1
        self.coef1 = torch.sqrt(alphas) * (1 - alpha_prev) / (1 - alpha_bars)
        self.coef2 = torch.sqrt(alpha_prev) * self.betas / (1 - alpha_bars)

    def sample_forward(self, x, t, eps=None):
        """
        正向过程方法

        self.alpha_bars是一个一维Tensor。而在并行训练中，我们一般会令t为一个形状为(batch_size, )的Tensor。
        PyTorch允许我们直接用self.alpha_bars[t]从self.alpha_bars里取出batch_size个数，
        就像用一个普通的整型索引来从数组中取出一个数一样。有些实现会用torch.gather从self.alpha_bars里取数，其作用是一样的。

        :param x:
        :param t:
        :param eps:
        :return:
        """
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        if eps is None:
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res

    def sample_backward(self, img_shape, net, device, simple_var=True, clip_x0=True):
        """
        在反向过程中，DDPM会用神经网络预测每一轮去噪的均值，把xt复原回x0，以完成图像生成。

        sample_backward是用来给外部调用的方法
        sample_backward_step是执行一步反向过程的方法

        :param img_shape:
        :param net:
        :param device:
        :param simple_var:
        :param clip_x0:
        :return:
        """
        # 随机生成纯噪声,对应xt
        x = torch.randn(img_shape).to(device)
        net = net.to(device)
        # 令t = n_steps - 1 到 0
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var, clip_x0)
        return x

    def sample_backward_step(self, x_t, t, net, simple_var=True, clip_x0=True):

        # batch size
        n = x_t.shape[0]
        # 时间整数格式
        t_tensor = torch.tensor([t] * n, dtype=torch.long).to(x_t.device).unsqueeze(1)
        eps = net(x_t, t_tensor)

        # 处理反向过程公式中的方差项，t非零的时候算方差项。
        if t == 0:
            noise = 0
        else:
            # 控制选哪种取值方式
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * self.betas[t]
            # 获取方差后，我们再随机采样一个噪声，根据公式，得到方差项。
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)

        if clip_x0:
            x_0 = (x_t - torch.sqrt(1 - self.alpha_bars[t]) * eps) / torch.sqrt(self.alpha_bars[t])
            x_0 = torch.clip(x_0, -1, 1)
            mean = self.coef1[t] * x_t + self.coef2[t] * x_0
        else:
            mean = (x_t -
                    (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
                    eps) / torch.sqrt(self.alphas[t])
        x_t = mean + noise

        return x_t
