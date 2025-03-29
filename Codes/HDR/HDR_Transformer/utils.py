import os
import glob
import json
import logging
import math
import torch
from math import log10
import coloredlogs
import cv2
import imageio
import numpy as np


def list_all_files_sorted(folder_name, extension=""):
    return sorted(glob.glob(os.path.join(folder_name, "*" + extension)))

def list_all_files_sorted(folder_name, extension=""):
    return sorted(glob.glob(os.path.join(folder_name, "*" + extension)))

def read_expo_times(file_name):
    return np.power(2, np.loadtxt(file_name))

def read_images(file_names):
    imgs = []
    for img_str in file_names:
        img = cv2.imread(img_str, -1)

        # equivalent to im2single from Matlab
        img = img / 2 ** 16
        img = np.float32(img)
        img.clip(0, 1)
        imgs.append(img)
    return np.array(imgs)

def read_images_rgb(file_names):
    imgs = []
    for img_str in file_names:
        img = cv2.cvtColor(cv2.imread(img_str, -1), cv2.COLOR_BGR2RGB)
        # equivalent to im2single from Matlab
        img = img / 2 ** 16
        img = np.float32(img)
        img.clip(0, 1)
        imgs.append(img)
    return np.array(imgs)

def read_label(file_path, file_name):
    label = imageio.imread(os.path.join(file_path, file_name), 'hdr')
    label = np.array(label, dtype=np.float32)
    label = label[:, :, [2, 1, 0]]  ##cv2
    return label

def ldr_to_hdr(imgs, expo, gamma):
    return (imgs ** gamma) / (expo + 1e-8)

def psnr(x, target):
    sqrdErr = np.mean((x - target) ** 2)
    return 10 * log10(1/sqrdErr)

def tensor_torch(batch, check_on=True):
    if check_on:
        for k, v in batch.items():
            #print(v.type())
            batch[k] = v.cuda()  # Convert to PyTorch Tensor
    else:
        for k, v in batch.items():
            batch[k] = v.numpy()  # Convert back to NumPy array
    return batch



