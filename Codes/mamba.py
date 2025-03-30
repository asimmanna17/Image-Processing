import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.drop_path = nn.Identity() if drop == 0. else nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * expand)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.mamba(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, d_state=16, d_conv=4, expand=2, drop=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(dim=dim, d_state=d_state, d_conv=d_conv, expand=expand, drop=drop)
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

# Example usage
dim = 128  # Embedding dimension
depth = 4  # Number of Mamba blocks
seq_length = 256  # Sequence length
batch_size = 8  # Batch size

x = torch.randn(batch_size, seq_length, dim)  # Input tensor of shape (B, L, C)

layer = BasicLayer(dim=dim, depth=depth)
output = layer(x)
print("Output shape:", output.shape)  # Expected output shape: (8, 256, 128)
