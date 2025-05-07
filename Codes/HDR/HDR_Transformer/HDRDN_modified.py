import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from alignhead import *
from HDR_transformer import *


class MultiCrossAlign_head_atttrans_res1sepalign(nn.Module):
    def __init__(self, in_c, num_heads=4, dim_align=64):
        super(MultiCrossAlign_head_atttrans_res1sepalign, self).__init__()
        
        #Shallow feature
        self.conv_f1 = nn.Conv2d(in_c, dim_align, 3, 1, 1)
        self.conv_f2 = nn.Conv2d(in_c, dim_align, 3, 1, 1)
        self.conv_f3 = nn.Conv2d(in_c, dim_align, 3, 1, 1)

        # Extract multi-scale feature
        self.pyramid1 = Pyramid(in_channels=dim_align, n_feats=dim_align)
        self.pyramid2 = Pyramid(in_channels=dim_align, n_feats=dim_align)
        self.pyramid3 = Pyramid(in_channels=dim_align, n_feats=dim_align)

        self.align1 = Pyramid_CrossattAlign_Atttrans(scales=3, num_feats=dim_align, num_heads=4, window_size=8)
        self.align2 = Pyramid_CrossattAlign_Atttrans(scales=3, num_feats=dim_align, num_heads=4, window_size=8)

    def forward(self, x1, x2, x3):
        # x1:sht  x2:mid   x3:lng
        H,W = x1.shape[2:]
        x1 = self.conv_f1(x1)
        x2 = self.conv_f2(x2)
        x3 = self.conv_f3(x3)
        
        x1_feature_list = self.pyramid1(x1)
        #print(x1_feature_list[2].shape)
        x2_feature_list = self.pyramid2(x2)
        x3_feature_list = self.pyramid3(x3)

        aligned_x1 = self.align1(x2_feature_list, x1_feature_list, patch_ratio_list=[2,2,2])
        aligned_x3 = self.align2(x2_feature_list, x3_feature_list, patch_ratio_list=[2,2,2]) 
        
        return [aligned_x1, x2, aligned_x3, x1, x3]


class SpatialAttentionModule(nn.Module): ### for fusion

    def __init__(self, dim):
        super().__init__()
        self.att1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        f_cat = torch.cat((x1, x2), 1)
        att_map = self.sigmoid(self.att2(self.relu(self.att1(f_cat))))
        return att_map

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg_pool(x))
        return x * w


class SpatialAttention(nn.Module):  #for HDR generation
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        avg = torch.mean(x, 1, keepdim=True)
        max_, _ = torch.max(x, 1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))
        return x * scale


class ExposureAwareRefiner(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.ca(x)
        x = self.sa(x)
        x = self.conv2(x)
        return x + residual


class HDRRefineFormer(nn.Module):
    def __init__(self, embed_dim=60, out_channels=4, depth=4):
        super().__init__()
        self.refine_blocks = nn.Sequential(*[
            ExposureAwareRefiner(embed_dim) for _ in range(depth)
        ])
        self.final = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, out_channels, 3, padding=1)
        )

    def forward(self, x):  # x: (B, 60, H, W)
        x = self.refine_blocks(x)
        return self.final(x)  # (B, 4, H, W)



class HDRDN_updated(nn.Module):
    def __init__(self, n_channel =8, out_channel = 4, embed_dim = 60):
        super(HDRDN_updated, self).__init__()
        self.in_channel = n_channel
        self.out_channel = out_channel
        embed_dim = embed_dim
        self.embed_dim = embed_dim
        #################### 1. Pyramid Cross-Attention Alignment Module #############################################
        self.align_head = MultiCrossAlign_head_atttrans_res1sepalign(in_c=self.in_channel, dim_align=self.embed_dim)
        #################### 2. Spatial Attention Module #############################################################
        self.att1 = SpatialAttentionModule(self.embed_dim)
        self.att2 = SpatialAttentionModule(self.embed_dim)
        self.conv_first = nn.Conv2d(self.embed_dim *3, self.embed_dim, 3, 1, 1)
        ################### 3. Build HDR blocl #######################################################################
        self.hdr_block = HDRRefineFormer(self.embed_dim, self.out_channel, depth=4)
        
        # build the last conv layer
        
        self.act_last = nn.Sigmoid()
        self.apply(self._init_weights)

    def forward(self, x1, x2, x3):
        f1_att, f2, f3_att, f1, f3 = self.align_head(x1, x2, x3)
        f1_att = f2 + f1_att
        f3_att = f2 + f3_att

        f1_att = self.att1(f1_att, f2)*f1_att
        f3_att = self.att2(f3_att, f2)*f3_att

        x = self.conv_first(torch.cat((f1_att, f2, f3_att), axis=1))

        x = self.hdr_block(x)
        
        output = self.act_last(x)
        return output

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

   

# Test the block
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_model = HDRDN_updated(n_channel =8, out_channel = 4, embed_dim = 60)
    full_model = full_model.cuda()
    npypath = '/data/asim/ISP/HDR_transformer/data/RAW/raw-2022-0606-2151-4147.npz'
    imdata = np.load(npypath)

    sht = imdata['sht']
    mid = imdata['mid']
    lng = imdata['lng']
    hdr = imdata['hdr']

    crop_size = 128
    H, W = hdr.shape[1], hdr.shape[2]
    start_h = (H - crop_size) // 2
    start_w = (W - crop_size) // 2

    sht_crop = sht[:, start_h:start_h+crop_size, start_w:start_w+crop_size]
    mid_crop = mid[:, start_h:start_h+crop_size, start_w:start_w+crop_size]
    lng_crop = lng[:, start_h:start_h+crop_size, start_w:start_w+crop_size]
    hdr_crop = hdr[:, start_h:start_h+crop_size, start_w:start_w+crop_size]

    #print(sht_crop.shape, hdr_crop.shape)

    def to_tensor(np_array):
        t = torch.from_numpy(np_array).float()
        return t

    im1 = to_tensor(sht_crop).to(device).unsqueeze(0)
    im2 = to_tensor(mid_crop).to(device).unsqueeze(0)
    im3 = to_tensor(lng_crop).to(device).unsqueeze(0)
    ref_hdr = to_tensor(hdr_crop).to(device).unsqueeze(0)
    with torch.no_grad():
        generate_hdr = full_model(im1, im2, im3)
    print("Output shape:", generate_hdr.shape)  