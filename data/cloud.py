import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import glob
from .transforms import *


def load_cloud(root):
    file_names = []
    for file_name in glob.glob(root+'/*'):
        file_names.append(file_name)
    file_names = sorted(file_names)
    length = len(file_names)
    return file_names, length


class Cloud(data.Dataset):
    def __init__(
            self, root, is_train=True, n_frames_input=5, n_frames_output=5,
            modality='L', transform=None):
        super(Cloud, self).__init__()
        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.modality = modality
        # self.video_dir is a list for training(or test) video dir, each video contains 10 images
        if self.is_train:
            if transform == None:
                self.transform = torchvision.transforms.Compose(
                    [GroupRandomHorizontalFlip(),
                     GroupResized((128, 192)),
                        Stack(),
                        ToTorchFormatTensor(),
                        GroupNormalize(mean=[0.0], std=[1.0])])
            self.video_dir, self.length = load_cloud(root+'train_newest_large')
        else:
            if transform == None:
                self.transform = torchvision.transforms.Compose(
                    [GroupResized((128, 192)),
                     Stack(),
                        ToTorchFormatTensor(),
                        GroupNormalize(mean=[0.0], std=[1.0])])

            self.video_dir, self.length = load_cloud(root+'test_newest_large')
        # self.image_shape = [128, 128]  # h,w
        self.dataset = None

    def read_images(self, images_dir):
        # data = np.zeros((self.n_frames_total, self.image_shape[0], self.image_shape[1], c),dtype=np.float32)
        data = list()
        images = glob.glob(images_dir+'/*')
        images = sorted(images)
        if len(images) < self.n_frames_total:
            # print(images)
            raise Exception("video not long enough!")
        for j, image in enumerate(images, 0):
            if j+1 > self.n_frames_total:
                break
            # data[j, :, :, :] = cv2.resize(cv2.imread(image), (self.image_shape[1], self.image_shape[0])).astype(
                # np.float32)  # without resize cuz the size of picture is [w,h]

            data.extend([Image.open(image).convert(self.modality)])
        return data

    def __getitem__(self, idx):
        if self.dataset is not None:
            images = self.dataset[:, idx, ...]
        else:
            images_dir = self.video_dir[idx]

        images = self.read_images(images_dir)
        if self.transform is not None:
            images = self.transform(images)
        images = images.view(self.n_frames_total, -1, images.shape[1], images.shape[2])
        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:self.n_frames_total]
        else:
            output = []

        out = [idx, input, output]
        return out

    def __len__(self):
        return self.length
