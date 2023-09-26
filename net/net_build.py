# -*- coding: utf-8 -*-
# @Time    : 2023/8/10 13:49
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12

from net.ConvNet import ConvNet
from net.UNet import UNet


def build_network(config: dict, n_steps):
    network_type = config.pop('type')
    if network_type == 'ConvNet':
        network_cls = ConvNet
    elif network_type == 'UNet':
        network_cls = UNet
    else:
        raise "None net type"

    network = network_cls(n_steps, **config)
    return network
