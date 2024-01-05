# -*- coding: utf-8 -*-
# @Time    : 2023/8/10 13:51
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
import torch
import torch.nn as nn


def get_img_shape(c=1,h=28,w=28):
    return (c, h, w)


class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()

        # Assume d_model is an even number for convenience
        assert d_model % 2 == 0

        pe = torch.zeros(max_seq_len, d_model)
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        pe_2i = torch.sin(pos / 10000 ** (two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000 ** (two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)

        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding.weight.data = pe
        self.embedding.requires_grad_(False)

    def forward(self, t):
        return self.embedding(t)


class ResidualBlock(nn.Module):

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.actvation1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.actvation2 = nn.ReLU()
        
        if in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1), nn.BatchNorm2d(out_c))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.actvation1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(x)
        x = self.actvation2(x)
        return x
