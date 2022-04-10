# -*- coding:utf-8 -*-
###
# File: main.py
# Created Date: Saturday, April 9th 2022, 10:45:31 am
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
import torch.nn as nn
import torch.nn.functional as F
import tensorboardX
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.metrics import structural_similarity as ssim
from mim import MIM
import random
import time
from data import *
import argparse
import os
import cv2
from os import system

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str,
                    default='mnist', help='dataset name')
parser.add_argument('--root', type=str,
                    default='/home/oulin/Dataset/cloud/', help='folder for dataset')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
# parser.add_argument(
#     '--input_shape', nargs='+', type=int,
#     help='input shape of the rnn')
# parser.add_argument('--input_dim', type=int, default=64, help='input dim of rnn')
# parser.add_argument(
#     '--hidden_dims', nargs='+', type=int,
#     help='hidden_dims of convlstm')
parser.add_argument('--n_epochs', type=int, default=500, help='nb of epochs')
parser.add_argument('--print_every', type=int, default=1, help='')
parser.add_argument('--eval_every', type=int, default=50, help='')
parser.add_argument('--save_every', type=int, default=50, help='')
parser.add_argument('--out_dir', type=str, default='./checkpoint/taxibj/',
                    help='models is saved in this dir')
# parser.add_argument('--img_dir', type=str, default='./results/taxibj/',
#                     help='images is saved in this dir')
parser.add_argument('--K', type=int, default=4, help='num of input frames')
parser.add_argument('--T', type=int, default=4, help='num of output frames')
parser.add_argument('--pre_trained', type=str, default=None, help='checkpoint dir')

args = parser.parse_args()
writer = SummaryWriter()

data_map = {'taxibj': Taxibj, 'cloud': Cloud, 'mnist': MovingMNIST, 'human': Human}

train_data = data_map[args.dataname](
    root=args.root, n_frames_input=args.K, n_frames_output=args.T)
train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=1,
    drop_last=True)

test_data = data_map[args.dataname](
    root=args.root, is_train=False, n_frames_input=args.K, n_frames_output=args.T)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data, batch_size=1, shuffle=False, num_workers=1)


def train_on_batch(
        input_tensor, target_tensor, encoder, encoder_optimizer, criterions,
        teacher_forcing_ratio):
    encoder_optimizer.zero_grad()
    # input_tensor : torch.Size([batch_size, input_length, 1, 64, 64])
    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)
    loss = 0
    network_input = torch.cat([input_tensor, target_tensor],
                              dim=1)
    gen_imgs = encoder(network_input,
                       teacher_forcing_ratio)
    for criterion in criterions:
        loss += criterion(gen_imgs*255, network_input[:, 1:]*255)
    loss.backward()
    encoder_optimizer.step()
    return loss.item()


def trainIters(encoder, n_epochs, print_every, eval_every, save_every):
    start = time.time()
    train_losses = []
    best_mse = float('inf')

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.0001)
    start_epoch = 0
    if args.pre_trained is not None:
        encoder_ckpt = torch.load(args.pre_trained)
        encoder.load_state_dict(encoder_ckpt['state_dict'])
        encoder_optimizer.load_state_dict(encoder_ckpt['optimizer'])
        start_epoch = encoder_ckpt['epoch']
        print('load pre_trained model successfully!')
    scheduler_enc = ReduceLROnPlateau(
        encoder_optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    criterions = [nn.MSELoss(), nn.L1Loss()]

    length = len(train_loader)
    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()
        loss_epoch = 0
        teacher_forcing_ratio = np.maximum(0, 1 - epoch * 0.01)

        for i, out in enumerate(train_loader, 0):
            #input_batch =  torch.Size([8, 20, 1, 64, 64])
            input_tensor = out[1].to(device)
            target_tensor = out[2].to(device)
            loss = train_on_batch(
                input_tensor, target_tensor, encoder, encoder_optimizer, criterions,
                teacher_forcing_ratio)
            loss_epoch += loss
            writer.add_scalar('loss', loss, i + epoch * length)

            if (i % 10) == 0:
                print(
                    'epoch: %d, iter: %d/%d, loss= %0.6f' %
                    (epoch, i, length, loss))

        train_losses.append(loss_epoch)
        if (epoch+1) % print_every == 0:
            print('epoch ', epoch,  ' loss ', loss_epoch, ' epoch time ', time.time()-t0)

        if (epoch+1) % eval_every == 0:
            mse, mae, ssim = evaluate(encoder, test_loader, epoch+1)
            writer.add_scalar('eval_mse', mse, epoch)
            # writer.add_scalar('accu_mean', accus[1].mean(), epoch)  # 0.05
            scheduler_enc.step(mse)
        if (epoch + 1) % save_every == 0:
            if not os.path.exists(args.out_dir):
                os.makedirs(args.out_dir)
            torch.save({'epoch': epoch + 1, 'state_dict': encoder.state_dict(),
                        'optimizer': encoder_optimizer.state_dict()},
                       os.path.join(args.out_dir, 'model_%d.pkl' % epoch))

    return train_losses


def cacul_accu(target, prediction, norms=[0.01, 0.05, 0.1]):
    """
    # Parameters
    # ----------
    # target_shape:(batch_size,5, 1, 480, 720) 

    # prediction_shape:(batch_size,5, 1, 480, 720)

    # all values are between 0 and 1
    """
    num = target.shape[0] * target.shape[2] * target.shape[3] * target.shape[4]
    accus = []
    for norm in norms:
        accu = np.sum(np.abs(target - prediction) < norm,
                      axis=(0, 2, 3, 4)) / num  # accu.shape (5,)
        accus.append(accu)
    accus = np.stack(accus)  # accus.shape (3,5) dtype: np.array
    return accus


def visualize(epoch, index, gt, pre, root, K=5, T=10):
    save_path = root + '%d/%d' % (epoch, index)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    gt = np.transpose(gt*255.0, (0, 3, 4, 2, 1))
    gt = gt.reshape(-1, gt.shape[1], gt.shape[2],
                    gt.shape[3], gt.shape[4]).astype(np.uint8)
    pre = np.transpose(pre*255.0, (0, 3, 4, 2, 1))
    pre = pre.reshape(-1, pre.shape[1], pre.shape[2],
                      pre.shape[3], pre.shape[4]).astype(np.uint8)

    for t in range(K+T):
        pre_each = pre[0, :, :, :, t]
        gt_each = gt[0, :, :, :, t]
        # pre_each = draw_frame(pre_each, t < K)
        # gt_each = draw_frame(gt_each, t < K)
        cv2.imwrite(save_path + '/pre_' + '{0:04d}'.format(t) + '.png', pre_each)
        cv2.imwrite(save_path+"/gt_"+"{0:04d}".format(t)+".png", gt_each)
    return 0


def evaluate(encoder, loader, epoch):
    total_mse, total_mae, total_ssim = 0, 0, 0
    with torch.no_grad():
        for i, out in enumerate(loader, 0):
            #input_batch = torch.Size([8, 20, 1, 64, 64])
            input_tensor = out[1].to(device)
            target_tensor = out[2].to(device)

            input_length = input_tensor.size()[1]
            target_length = target_tensor.size()[1]

            network_input = torch.cat([input_tensor, target_tensor],
                                      dim=1)
            gen_imgs = encoder(network_input)  # [bs, T, C, h, w]

            input = input_tensor.cpu().numpy()
            target = target_tensor.cpu().numpy()
            predictions = gen_imgs[:, -args.T:].cpu().numpy()
            # predictions = predictions.swapaxes(1, 2)  # (batch_size,10, 1, 64, 64)

            # visualize video
            # gt = np.concatenate((input, target), axis=1)
            # pre = np.concatenate((input, predictions), axis=1)
            # visualize(epoch, i, gt, pre, args.img_dir, args.K, args.T)

            # add quantitative evaluation norm
            # accus_batch = cacul_accu(target, predictions)
            mse_batch = np.mean((predictions-target)**2, axis=(0, 1, 2)).sum()
            mae_batch = np.mean(np.abs(predictions-target),  axis=(0, 1, 2)).sum()

            # total_accus += accus_batch
            total_mse += mse_batch
            total_mae += mae_batch

            for a in range(0, target.shape[0]):
                for b in range(0, target.shape[1]):
                    total_ssim += ssim(target[a, b, 0, ], predictions[a,
                                       b, 0, ]) / (target.shape[0]*target.shape[1])

            print('visualizing the %dth video' % i)
            # cross_entropy = -target*np.log(predictions) - (
            #     1-target) * np.log(1-predictions)
            # cross_entropy = cross_entropy.sum()
            # cross_entropy = cross_entropy / (args.batch_size*target_length)
            # total_bce += cross_entropy

    print('eval mse ', total_mse / len(loader),
          ' eval mae ', total_mae / len(loader),
          ' eval ssim ', total_ssim / len(loader))
    return total_mse/len(loader),  total_mae/len(loader), total_ssim/len(loader)


print('BEGIN TRAIN')
# input shape depends on the original images, if dataset is MNIST, input shape should be (16,16), if dataset is cloud, input shape should be (32,32)
# phycell = PhyCell(
#     input_shape=args.input_shape,
#     input_dim=args.input_dim, F_hidden_dims=[49],
#     n_layers=1, kernel_size=(7, 7),
#     device=device)
# convlstm = ConvLSTM(
#     input_shape=args.input_shape,
#     input_dim=args.input_dim, hidden_dims=args.hidden_dims,
#     n_layers=3, kernel_size=(3, 3),
#     device=device)
encoder = MIM(input_dims=2, out_dims=2,
              in_shape=[args.batch_size, 2, 32, 32],
              hidden_dim=[64, 64, 64, 64],
              total_length=args.K + args.T, input_length=args.K, device=device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('encoder ', count_parameters(encoder))

plot_losses = trainIters(
    encoder, args.n_epochs, print_every=args.print_every, eval_every=args.eval_every,
    save_every=args.save_every)
# print(plot_losses)
