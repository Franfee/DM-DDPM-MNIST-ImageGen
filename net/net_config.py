# -*- coding: utf-8 -*-
# @Time    : 2023/8/10 13:43
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12

convnet_small_cfg = {'type': 'ConvNet', 'intermediate_channels': [10, 20], 'pe_dim': 128}

convnet_medium_cfg = {'type': 'ConvNet',
                      'intermediate_channels': [10, 10, 20, 20, 40, 40, 80, 80],
                      'pe_dim': 256,
                      'insert_t_to_all_layers': True
                      }
convnet_big_cfg = {'type': 'ConvNet',
                   'intermediate_channels': [20, 20, 40, 40, 80, 80, 160, 160],
                   'pe_dim': 256,
                   'insert_t_to_all_layers': True
                   }

unet_1_cfg = {'type': 'UNet', 'channels': [10, 20, 40, 80], 'pe_dim': 128}

unet_res_cfg = {'type': 'UNet', 'channels': [10, 20, 40, 80], 'pe_dim': 128, 'residual': True}

configs = [
    convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg, unet_res_cfg
]
