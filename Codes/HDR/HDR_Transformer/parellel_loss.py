import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import ms_ssim
import numpy as np
import cv2
import math

# Assume netA, netB, and netC are defined elsewhere
netA = NetworkA().to(device)
netB = NetworkB().to(device)
netC = NetworkC().to(device)

optA = optim.Adam(netA.parameters(), lr=1e-4)
optB = optim.Adam(netB.parameters(), lr=1e-4)
optC = optim.Adam(netC.parameters(), lr=1e-4)

loss_fn_l1 = nn.L1Loss()

# Differentiable Bayer-to-RGB function using bilinear interpolation
def bayer_to_rgb(bayer_array):
    # Ensure the input is a PyTorch tensor and move to CPU for NumPy operations
    bayer_array = bayer_array.detach().cpu().numpy()

    H, W = bayer_array.shape[1:]
    bayer = np.zeros((H * 2, W * 2), dtype=bayer_array.dtype)

    # Interpolate Bayer channels
    bayer[0::2, 0::2] = bayer_array[0]  # Red channel
    bayer[0::2, 1::2] = bayer_array[1]  # Green channel (odd rows)
    bayer[1::2, 0::2] = bayer_array[2]  # Green channel (even rows)
    bayer[1::2, 1::2] = bayer_array[3]  # Blue channel

    # Clip and scale to 16-bit range
    bayer = np.clip(bayer, 0, 1)
    bayer = (bayer * 65535).astype(np.uint16)

    # Perform demosaicing using OpenCV
    rgb = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB)

    # Normalize to [0, 1] range
    rgb = rgb.astype(np.float32) / 65535.0

    # Normalize to maximum brightness
    if np.max(rgb) > 0:
        rgb = rgb / np.max(rgb)

    # Apply gamma correction (optional)
    rgb = np.clip(rgb, 0, 1) ** (1 / 2.2)

    # Convert back to a PyTorch tensor and move to the same device as input
    return torch.tensor(rgb).to(bayer_array.device)

# Range compressor instead of tonemap
def range_compressor(hdr_img, mu=5000):
    return torch.log(1 + mu * hdr_img) / math.log(1 + mu)

# Phase correlation loss
def phase_correlation_loss(pred_img, gt_img):
    loss_list = []
    pred_img = pred_img.detach().cpu().numpy()
    gt_img = gt_img.detach().cpu().numpy()
    for b in range(pred_img.shape[0]):
        pred_gray = np.mean(pred_img[b], axis=0).astype(np.float32)
        gt_gray = np.mean(gt_img[b], axis=0).astype(np.float32)
        shift, _ = cv2.phaseCorrelate(pred_gray, gt_gray)
        shift_mag = np.sqrt(shift[0]**2 + shift[1]**2)
        loss_list.append(shift_mag)
    return torch.tensor(loss_list).mean()

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        input_img = batch["input"].to(device)
        target_raw = batch["target_raw"].to(device)
        target_rgb = batch["target_rgb"].to(device)

        # Forward pass through netA
        outputA1, outputA2 = netA(input_img)

        # Loss1: range-compressed outputA1 and target_raw
        compressed_A1 = range_compressor(outputA1)
        compressed_target = range_compressor(target_raw)
        loss1 = loss_fn_l1(compressed_A1, compressed_target)

        # Loss2: MS-SSIM on RGB image
        rgb_input = bayer_to_rgb(outputA1)
        outputB = netB(rgb_input)
        loss2 = 1 - ms_ssim(outputB, target_rgb, data_range=1.0, size_average=True)

        # Loss3: Phase correlation loss
        outputC = netC(outputA2)
        loss3 = phase_correlation_loss(outputC, target_raw)

        # Total loss
        total_loss = loss1 + loss2 + loss3

        # Backpropagation
        optA.zero_grad()
        optB.zero_grad()
        optC.zero_grad()
        total_loss.backward()
        optA.step()
        optB.step()
        optC.step()
