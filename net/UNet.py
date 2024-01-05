# -*- coding: utf-8 -*-
# @Time    : 2023/8/10 11:54
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.base import PositionalEncoding, get_img_shape


class UnetBlock(nn.Module):

    def __init__(self, shape, in_c, out_c, residual=False):
        super().__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.activation = nn.ReLU()
        self.residual = residual
        if residual:
            if in_c == out_c:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        out = self.ln(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.residual:
            out += self.residual_conv(x)
        out = self.activation(out)
        return out


class UNet(nn.Module):

    def __init__(self, n_steps, channels=[10, 20, 40, 80], pe_dim=128, residual=True) -> None:
        super().__init__()
        C, H, W = get_img_shape()
        layers = len(channels)
        Hs = [H]
        Ws = [W]
        cH = H
        cW = W
        for _ in range(layers - 1):
            cH //= 2
            cW //= 2
            Hs.append(cH)
            Ws.append(cW)

        self.pe = PositionalEncoding(n_steps, pe_dim)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pe_linears_en = nn.ModuleList()
        self.pe_linears_de = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        prev_channel = C
        for channel, cH, cW in zip(channels[0:-1], Hs[0:-1], Ws[0:-1]):
            self.pe_linears_en.append(
                nn.Sequential(nn.Linear(pe_dim, prev_channel), nn.ReLU(),
                              nn.Linear(prev_channel, prev_channel)))
            self.encoders.append(
                nn.Sequential(
                    UnetBlock((prev_channel, cH, cW),
                              prev_channel,
                              channel,
                              residual=residual),
                    UnetBlock((channel, cH, cW),
                              channel,
                              channel,
                              residual=residual)))
            self.downs.append(nn.Conv2d(channel, channel, 2, 2))
            prev_channel = channel

        self.pe_mid = nn.Linear(pe_dim, prev_channel)
        channel = channels[-1]
        self.mid = nn.Sequential(
            UnetBlock((prev_channel, Hs[-1], Ws[-1]),
                      prev_channel,
                      channel,
                      residual=residual),
            UnetBlock((channel, Hs[-1], Ws[-1]),
                      channel,
                      channel,
                      residual=residual),
        )
        prev_channel = channel
        for channel, cH, cW in zip(channels[-2::-1], Hs[-2::-1], Ws[-2::-1]):
            self.pe_linears_de.append(nn.Linear(pe_dim, prev_channel))
            self.ups.append(nn.ConvTranspose2d(prev_channel, channel, 2, 2))
            self.decoders.append(
                nn.Sequential(
                    UnetBlock((channel * 2, cH, cW),
                              channel * 2,
                              channel,
                              residual=residual),
                    UnetBlock((channel, cH, cW),
                              channel,
                              channel,
                              residual=residual)))

            prev_channel = channel

        self.conv_out = nn.Conv2d(prev_channel, C, 3, 1, 1)

    def forward(self, x, t):
        n = t.shape[0]
        t = self.pe(t)
        encoder_outs = []
        for pe_linear, encoder, down in zip(self.pe_linears_en, self.encoders,
                                            self.downs):
            pe = pe_linear(t).reshape(n, -1, 1, 1)
            x = encoder(x + pe)
            encoder_outs.append(x)
            x = down(x)
        pe = self.pe_mid(t).reshape(n, -1, 1, 1)
        x = self.mid(x + pe)
        for pe_linear, decoder, up, encoder_out in zip(self.pe_linears_de,
                                                       self.decoders, self.ups,
                                                       encoder_outs[::-1]):
            pe = pe_linear(t).reshape(n, -1, 1, 1)
            x = up(x)

            pad_x = encoder_out.shape[2] - x.shape[2]
            pad_y = encoder_out.shape[3] - x.shape[3]
            x = F.pad(x, (pad_x // 2, pad_x - pad_x // 2, pad_y // 2,
                          pad_y - pad_y // 2))
            x = torch.cat((encoder_out, x), dim=1)
            x = decoder(x + pe)
        x = self.conv_out(x)
        return x
