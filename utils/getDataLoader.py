# -*- coding: utf-8 -*-
# @Time    : 2023/8/10 11:26
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12

import os
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda

BASE_DIR = os.getcwd()
if "utils" not in BASE_DIR:
    BASE_DIR = os.path.join(BASE_DIR, "utils")


def get_dataloader(batch_size: int, train: bool):
    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])

    dataset = torchvision.datasets.MNIST(root=os.path.join(BASE_DIR, "..", "datasets"),
                                         train=train,
                                         transform=transform,
                                         download=True)
    
    return DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)


if __name__ == '__main__':

    dl = get_dataloader(batch_size=4, train=True)
    for idx, (batch_data) in enumerate(dl):
        if idx > 4:
            break

        img, label = batch_data
        print(img.shape)
        print(label.shape)
