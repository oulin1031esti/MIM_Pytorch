# -*- coding:utf-8 -*-
###
# File: human.py
# Created Date: Wednesday, April 6th 2022, 3:24:56 pm
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
import cv2
import os


class Human(data.Dataset):
    def __init__(
            self, root, is_train=True, n_frames_input=4, n_frames_output=4,
            transform=None):
        super(Human, self).__init__()
        self.root = root
        self.is_train = is_train
        self.nframe_in = n_frames_input
        self.nframe_out = n_frames_output
        self.seq_len = self.nframe_in+self.nframe_out
        self.intervel = 2
        self.transform = transform
        self.image_width = 128
        self.datasets, self.indices = self.load_data()

    def load_data(self):
        intervel = 2
        frames_np = []
        scenarios = ['Walking']
        if self.is_train:
            subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
        else:
            subjects = ['S9', 'S11']
        print('load data...', self.root)
        filenames = os.listdir(self.root)
        filenames.sort()
        print('data size', len(filenames))
        frames_file_name = []
        for filename in filenames:
            fix = filename.split('.')
            fix = fix[0]
            subject = fix.split('_')
            scenario = subject[1]
            subject = subject[0]
            if subject not in subjects or scenario not in scenarios:
                continue
            file_path = os.path.join(self.root, filename)
            image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
            # [1000,1000,3]
            image = image[image.shape[0]//4:-image.shape[0] //
                          4, image.shape[1]//4:-image.shape[1]//4, :]
            if self.image_width != image.shape[0]:
                image = cv2.resize(image, (self.image_width, self.image_width))
            # image = cv2.resize(image[100:-100,100:-100,:], (self.image_width, self.image_width),
            #                   interpolation=cv2.INTER_LINEAR)
            frames_np.append(np.array(image, dtype=np.float32) / 255.0)
            frames_file_name.append(filename)
            indices = []

        index = 0
        print('gen index')
        while index + intervel * self.seq_len - 1 < len(frames_file_name):
            # 'S11_Discussion_1.54138969_000471.jpg'
            # ['S11_Discussion_1', '54138969_000471', 'jpg']
            start_infos = frames_file_name[index].split('.')
            end_infos = frames_file_name[index+intervel*(self.seq_len-1)].split('.')
            if start_infos[0] != end_infos[0]:
                index += 1
                continue
            start_video_id, start_frame_id = start_infos[1].split('_')
            end_video_id, end_frame_id = end_infos[1].split('_')
            if start_video_id != end_video_id:
                index += 1
                continue
            if int(end_frame_id) - int(start_frame_id) == 5 * (self.seq_len - 1) * intervel:
                indices.append(index)
            if self.is_train:
                index += 10
            else:
                index += 5

        print("there are " + str(len(indices)) + " sequences")
        # data = np.asarray(frames_np)
        data = frames_np
        print("there are " + str(len(data)) + " pictures")
        return data, indices

    def __getitem__(self, idx):
        begin = self.indices[idx]
        end = begin + self.seq_len * self.intervel

        images = self.datasets[begin:end:self.intervel]
        images = np.stack(images, axis=0)  # (8,128,128,3)
        assert images.shape[0] == self.seq_len
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
        return len(self.indices)
