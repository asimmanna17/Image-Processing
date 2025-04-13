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


class SpatialAttentionModule(nn.Module):

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



class Alignment(nn.Module):
    def __init__(self, n_channel =8, out_channel = 4, embed_dim = 60, depths=[4, 4, 4]):
        super(Alignment, self).__init__()
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
        
        self.apply(self._init_weights)

    def forward(self, x1, x2, x3):
        f1_att, f2, f3_att, f1, f3 = self.align_head(x1, x2, x3)
        f1_att = f2 + f1_att
        f3_att = f2 + f3_att

        f1_att = self.att1(f1_att, f2)*f1_att
        f3_att = self.att2(f3_att, f2)*f3_att

        x = self.conv_first(torch.cat((f1_att, f2, f3_att), axis=1))

        
        return x

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

    def forward_features(self, x):
        # x [B, embed_dim, h, w]
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x) # B L C
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)
        return x
