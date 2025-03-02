import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Transformer Block for Multi-Exposure Fusion
class TransformerFusion(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# Spatial Attention for Fusion
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.conv1(x))
        return x * attn

# Multi-Exposure Fusion Network
class MultiEVFusionNet(nn.Module):
    def __init__(self, img_dim=128, num_heads=4, num_exposures=3):
        super().__init__()
        self.transformer = TransformerFusion(dim=img_dim, num_heads=num_heads)
        self.spatial_attn = SpatialAttention(in_channels=num_exposures)
        self.conv_out = nn.Conv2d(num_exposures, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = rearrange(x, 'b e h w -> b (h w) e')  # Prepare for transformer
        x = self.transformer(x)
        x = rearrange(x, 'b (h w) e -> b e h w', h=int(x.shape[1]**0.5))
        x = self.spatial_attn(x)  # Spatial Attention for Fusion
        x = self.conv_out(x)  # Convert to single RAW image
        return x

# Instantiate Model
model = MultiEVFusionNet()
print(model)
