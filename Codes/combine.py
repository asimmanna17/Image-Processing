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

# Pyramid Cross-Attention Alignment Module
class PyramidCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, levels=3):
        super().__init__()
        self.levels = levels
        self.transformers = nn.ModuleList([
            TransformerFusion(dim, num_heads) for _ in range(levels)
        ])

    def forward(self, x):
        for i in range(self.levels):
            x = self.transformers[i](x)
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
        self.pyramid_attention = PyramidCrossAttention(img_dim, num_heads)
        self.spatial_attn = SpatialAttention(in_channels=num_exposures)
        self.conv_out = nn.Conv2d(num_exposures, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = rearrange(x, 'b e h w -> b (h w) e')  # Prepare for transformer
        x = self.pyramid_attention(x)
        x = rearrange(x, 'b (h w) e -> b e h w', h=int(x.shape[1]**0.5))
        x = self.spatial_attn(x)  # Spatial Attention for Fusion
        x = self.conv_out(x)  # Convert to single RAW image
        return x

# HDR Reconstruction using Context-Aware Transformer
class ContextAwareTransformer(nn.Module):
    def __init__(self, in_channels=3, dim=128, num_heads=4):
        super().__init__()
        self.transformer = TransformerFusion(dim, num_heads)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=int(x.shape[1]**0.5))
        x = self.conv3(x)
        return x

# Complete HDR Pipeline
class HDRPipeline(nn.Module):
    def __init__(self, img_dim=128, num_heads=4, num_exposures=3):
        super().__init__()
        self.fusion = MultiEVFusionNet(img_dim=img_dim, num_heads=num_heads, num_exposures=num_exposures)
        self.reconstruction = ContextAwareTransformer()

    def forward(self, x):
        x = self.fusion(x)  # Multi-exposure fusion to single RAW
        x = self.reconstruction(x)  # HDR Reconstruction using Transformer
        return x

# Instantiate Model
model = HDRPipeline()
print(model)
