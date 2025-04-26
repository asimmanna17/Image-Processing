import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import OrderedDict
from scipy.ndimage import shift as shift_fn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def bayer_to_rgb(bayer_array):
    H, W = bayer_array.shape[1:]
    bayer = np.zeros((H * 2, W * 2), dtype=bayer_array.dtype)
    bayer[0::2, 0::2] = bayer_array[0]
    bayer[0::2, 1::2] = bayer_array[1]
    bayer[1::2, 0::2] = bayer_array[2]
    bayer[1::2, 1::2] = bayer_array[3]

    bayer = np.clip(bayer, 0, 1)
    bayer = (bayer * 65535).astype(np.uint16)

    #print("Max value before demosaicing:", np.max(bayer))

    rgb = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB)

    rgb = rgb.astype(np.float32) / 65535.0

    #print("Max RGB value after demosaicing:", np.max(rgb))

    # --- Normalize to maximum brightness ---
    if np.max(rgb) > 0:
        rgb = rgb / np.max(rgb)

    # Optional: Apply gamma correction (makes image bright and natural)
    rgb = np.clip(rgb, 0, 1) ** (1/2.2)

    return rgb

def to_tensor(np_array):
    t = torch.from_numpy(np_array).float()
    return t

def data_load(npypath):
    imdata = np.load(npypath)

    #sht = imdata['sht']
    mid = imdata['mid']
    #lng = imdata['lng']
    hdr = imdata['hdr']

    mid = to_tensor(mid).to(device).unsqueeze(0)
    hdr = to_tensor(hdr).to(device).unsqueeze(0)
    
    return mid, hdr

def phase_correlation_shift(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    shift, _ = cv2.phaseCorrelate(img1, img2)
    return shift

def shift_adjustment(img, shift):
    a = round(shift[0])
    b = round(shift[1])
    shifted_gt = np.roll(np.roll(img, shift=-a, axis=1), shift = -b, axis=0)
    return shifted_gt

def shift_image_subpixel(img, shift):
    a = -shift[1]
    b = -shift[0]
    order = 1
    if img.ndim == 2:  # grayscale image, no channel
        shifted = shift_fn(img, shift=(a, b), order=order, mode='nearest')
    else:  # color image, has channels
        shifted = np.zeros_like(img)
        channel = img.shape[2]
        for c in range(channel):
            shifted[:, :, c] = shift_fn(img[:, :, c], shift=(a, b), order=order, mode='nearest')
    
    return shifted


def global_shift(mid, gt):
    shifted_gt_list = []

    for b in range(gt.shape[0]):
        mid_numpy = mid[b][:4, :, :].squeeze().cpu().numpy()
        gt_numpy = gt[b].squeeze().cpu().numpy()

        mid_rgb = bayer_to_rgb(mid_numpy)
        gt_rgb = bayer_to_rgb(gt_numpy)

        initial_shift = phase_correlation_shift(mid_rgb, gt_rgb)
        shifted_gt_rgb = shift_image_subpixel(gt_rgb, initial_shift) #(H,W,C)
        after_shift = phase_correlation_shift(mid_rgb, shifted_gt_rgb)

        shifted_gt_tensor = to_tensor(shifted_gt_rgb).permute(2, 0, 1)  # Should be (C,H,W)
        shifted_gt_list.append(shifted_gt_tensor)

    shifted_gt_batch = torch.stack(shifted_gt_list, dim=0)  # (B, C, H, W)

    return shifted_gt_batch.to(device), initial_shift, after_shift

if __name__ == '__main__':
    npypath = '/data/asim/ISP/HDR_transformer/data/RAW/raw-2022-0606-2151-4147.npz'
    mid, gt = data_load(npypath)
    #print(mid.shape)
    #print(gt.shape)
    shifted_gt_rgb, intial_shift, after_shift = global_shift(mid, gt)
    print(shifted_gt_rgb.shape)
    print(intial_shift)
    print(after_shift)