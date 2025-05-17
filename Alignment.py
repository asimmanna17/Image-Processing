import torch
import torch.nn as nn

class MultiCrossAlign_head_atttrans_res1sepalign(nn.Module):
    def __init__(self, in_c, num_heads=4, dim_align=64):
        super(MultiCrossAlign_head_atttrans_res1sepalign, self).__init__()
        
        # Shallow features
        self.conv_f1 = nn.Conv2d(in_c, dim_align, 3, 1, 1)
        self.conv_f2 = nn.Conv2d(in_c, dim_align, 3, 1, 1)
        self.conv_f3 = nn.Conv2d(in_c, dim_align, 3, 1, 1)

        # Reliability-aware gates before Pyramid
        self.gate1 = nn.Sequential(
            nn.Conv2d(dim_align * 2, dim_align, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_align, dim_align, 3, 1, 1),
            nn.Sigmoid()
        )

        self.gate3 = nn.Sequential(
            nn.Conv2d(dim_align * 2, dim_align, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_align, dim_align, 3, 1, 1),
            nn.Sigmoid()
        )

        # Extract multi-scale features
        self.pyramid1 = Pyramid(in_channels=dim_align, n_feats=dim_align)
        self.pyramid2 = Pyramid(in_channels=dim_align, n_feats=dim_align)
        self.pyramid3 = Pyramid(in_channels=dim_align, n_feats=dim_align)

        # Alignment modules
        self.align1 = Pyramid_CrossattAlign_Atttrans(scales=3, num_feats=dim_align, num_heads=num_heads, window_size=8)
        self.align2 = Pyramid_CrossattAlign_Atttrans(scales=3, num_feats=dim_align, num_heads=num_heads, window_size=8)

    def forward(self, x1, x2, x3):
        # x1: short, x2: mid, x3: long
        x1 = self.conv_f1(x1)
        x2 = self.conv_f2(x2)
        x3 = self.conv_f3(x3)

        # Gate x1 and x3 with x2 for dynamic reliability handling
        x1_g = self.gate1(torch.cat([x1, x2], dim=1))
        x3_g = self.gate3(torch.cat([x3, x2], dim=1))
        x1 = x1 * x1_g
        x3 = x3 * x3_g

        # Multi-scale feature extraction
        x1_feature_list = self.pyramid1(x1)
        x2_feature_list = self.pyramid2(x2)
        x3_feature_list = self.pyramid3(x3)

        # Alignment
        aligned_x1 = self.align1(x2_feature_list, x1_feature_list, patch_ratio_list=[2, 2, 2])
        aligned_x3 = self.align2(x2_feature_list, x3_feature_list, patch_ratio_list=[2, 2, 2])

        return [aligned_x1, x2, aligned_x3, x1, x3]
