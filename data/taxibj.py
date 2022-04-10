# -*- coding:utf-8 -*-
###
# File: taxibj.py
# Created Date: Wednesday, March 30th 2022, 9:57:46 am
# Author: Oulin
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2022 Oulin
#
# All shall be well and all shall be well and all manner of things shall be well.
# Nope...we're doomed!
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###
import numpy as np
import torch
import torch.utils.data as data


class Taxibj(data.Dataset):
    def __init__(
            self, root, is_train=True, n_frames_input=4, n_frames_output=4,
            transform=None):
        super(Taxibj, self).__init__()
        self.root = root
        self.is_train = is_train
        self.nframe_in = n_frames_input
        self.nframe_out = n_frames_output
        self.transform = transform
        if self.is_train:
            self.dataset = np.load(self.root+'train.npy')
        else:
            self.dataset = np.load(self.root+'test.npy')

    def __getitem__(self, idx):
        images = self.dataset[idx]  # (8,32,32,2)
        if self.transform is not None:
            images = self.transform(images)

        images = images.transpose(0, 3, 1, 2)
        input = images[:self.nframe_in]
        output = images[self.nframe_in:]

        output = torch.from_numpy(output).contiguous().float()
        input = torch.from_numpy(input).contiguous().float()
        out = [idx, input, output]
        return out

    def __len__(self):
        return self.dataset.shape[0]
