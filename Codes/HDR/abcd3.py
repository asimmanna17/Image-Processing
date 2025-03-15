import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms.functional import rgb_to_grayscale

# --------------------------------------
# 1. Dynamic Tanh Activation
# --------------------------------------
class DynamicTanh(nn.Module):
    def __init__(self, scale=1.0):
        super(DynamicTanh, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
    
    def forward(self, x):
        return self.scale * torch.tanh(x)

# --------------------------------------
# 2. Pyramid Feature Extraction
# --------------------------------------
class Pyramid(nn.Module):
    def __init__(self, in_channels, n_feats, kernel_sizes=[3,3,3], strides=[1,2,2], paddings=[1,1,1]):
        super(Pyramid, self).__init__()
        self.in_channels = in_channels
        self.n_feats = n_feats
        layers = []
        in_channel = self.in_channels
        for i in range(len(kernel_sizes)):
            cur_layer = nn.Sequential(
                nn.Conv2d(in_channel, self.n_feats, kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i]),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
            )
            layers.append(cur_layer)
            in_channel = self.n_feats
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        feature_list = []
        for layer in self.layers:
            x = layer(x)
            feature_list.append(x)
        return feature_list

# --------------------------------------
# 3. Pyramid Cross-Attention Alignment Transformer
# --------------------------------------
class Pyramid_CrossattAlign_Atttrans(nn.Module):
    def __init__(self, scales, num_feats, window_size, num_heads=1, attn=None):
        super(Pyramid_CrossattAlign_Atttrans, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.scales = scales
        self.feat_conv = nn.ModuleDict()
        self.align = nn.ModuleDict()
        self.attn = attn if attn else nn.MultiheadAttention(embed_dim=num_feats, num_heads=num_heads, batch_first=True)
        self.window_size = window_size
        self.num_heads = num_heads
        self.layer_norm = nn.LayerNorm(num_feats)
        self.dynamic_tanh = DynamicTanh()
        
        for i in range(self.scales, 0, -1):
            level = f'l{i}'
            self.align[level] = nn.MultiheadAttention(embed_dim=num_feats, num_heads=num_heads, batch_first=True)
            if i < self.scales:
                self.feat_conv[level] = nn.Conv2d(num_feats*3, num_feats, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, ref_feats_list, toalign_feats_list, patch_ratio_list):
        upsample_feat = None
        last_att = None
        for i in range(self.scales, 0, -1):
            level = f'l{i}'
            ref_feat = ref_feats_list[i-1].permute(0,2,3,1)
            toalign_feat = toalign_feats_list[i-1].permute(0,2,3,1)
            aligned_feat, att = self.align[level](ref_feat.view(ref_feat.shape[0], -1, ref_feat.shape[-1]),
                                                  toalign_feat.view(toalign_feat.shape[0], -1, toalign_feat.shape[-1]),
                                                  toalign_feat.view(toalign_feat.shape[0], -1, toalign_feat.shape[-1]))
            aligned_feat = aligned_feat.view(ref_feat.shape).permute(0,3,1,2)
            aligned_feat = self.layer_norm(aligned_feat)
            aligned_feat = self.dynamic_tanh(aligned_feat)
            
            if i < self.scales:
                patch_ratio = patch_ratio_list[i-1]
                atttransfer_feat = self.feat_conv[level](torch.cat([aligned_feat, upsample_feat], dim=1))
                feat = atttransfer_feat
            else:
                feat = aligned_feat
            if i > 1:
                feat = self.lrelu(feat)
                upsample_feat = self.upsample(feat)
            last_att = att
        feat = self.lrelu(feat)
        return feat

# --------------------------------------
# 4. HDR Fusion Module
# --------------------------------------
class HDRFusionModule(nn.Module):
    def __init__(self, num_feats):
        super(HDRFusionModule, self).__init__()
        self.conv1 = nn.Conv2d(num_feats * 3, num_feats, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_feats, 3, kernel_size=3, padding=1)
    
    def forward(self, f1, f2, f3):
        fused = torch.cat([f1, f2, f3], dim=1)
        fused = self.relu(self.conv1(fused))
        return self.conv2(fused)

# --------------------------------------
# 5. Training with Random Data
# --------------------------------------
batch_size = 4
image_size = (3, 224, 224)  # RGB images of size 64x64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = HDRFusionModule(num_feats=64).to(device)

g_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))

for epoch in range(5):  # Run for fewer epochs just for verification
    l1 = torch.randn(batch_size, *image_size).to(device)
    l2 = torch.randn(batch_size, *image_size).to(device)
    l3 = torch.randn(batch_size, *image_size).to(device)
    
    fake_hdr = generator(l1, l2, l3)
    print(f"Epoch {epoch}: Generated HDR Shape: {fake_hdr.shape}")

