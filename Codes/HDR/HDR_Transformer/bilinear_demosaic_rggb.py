import torch
import torch.nn.functional as F

def bilinear_demosaic_rggb(bayer):  # [B, 1, H, W]
    B, _, H, W = bayer.shape

    r_mask = torch.zeros_like(bayer)
    g_mask = torch.zeros_like(bayer)
    b_mask = torch.zeros_like(bayer)

    # RGGB pattern
    r_mask[:, 0, 0::2, 0::2] = 1
    g_mask[:, 0, 0::2, 1::2] = 1
    g_mask[:, 0, 1::2, 0::2] = 1
    b_mask[:, 0, 1::2, 1::2] = 1

    r = F.avg_pool2d(bayer * r_mask, 3, 1, 1)
    g = F.avg_pool2d(bayer * g_mask, 3, 1, 1)
    b = F.avg_pool2d(bayer * b_mask, 3, 1, 1)

    rgb = torch.cat([r, g, b], dim=1)  # [B, 3, H, W]
    return rgb
