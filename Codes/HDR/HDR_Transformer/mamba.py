import torch
import torch.nn as nn
import torch.nn.functional as F

class Mamba(nn.Module):
    def __init__(self, in_dim, embed_dim=None, ssm_state='forward', ssm_init='s4d', dropout=0.1):
        super(Mamba, self).__init__()
        if embed_dim is None:
            embed_dim = in_dim
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.ssm_state = ssm_state

        self.init_scale = 0.1 if ssm_init == 's4d' else 1.0

        self.spatial_conv = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim)
        nn.init.normal_(self.spatial_conv.weight, std=self.init_scale)

        self.temporal_conv = nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim)
        nn.init.normal_(self.temporal_conv.weight, std=self.init_scale)

        self.gate = nn.Linear(in_dim, in_dim)
        self.proj = nn.Linear(in_dim, in_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        B, N, C = x.shape
        residual = x

        H = W = int(N ** 0.5)
        x_spatial = x.transpose(1, 2).reshape(B, C, H, W)
        x_spatial = self.spatial_conv(x_spatial)
        x_spatial = x_spatial.flatten(2).transpose(1, 2)

        x_temporal = x.transpose(1, 2)
        x_temporal = self.temporal_conv(x_temporal).transpose(1, 2)

        x = x_spatial + x_temporal

        gate = torch.sigmoid(self.gate(residual))
        x = gate * self.proj(x)
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x

class HDRMambaBlock(nn.Module):
    def __init__(self, embed_dim, seq_len, variant='v2', ssm_init='s4d'):
        super(HDRMambaBlock, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.pre_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU()
        )

        mamba_input_dim = embed_dim * 2

        if variant == 'v1':
            self.mamba = Mamba(in_dim=mamba_input_dim, embed_dim=embed_dim, ssm_state="bidirectional", ssm_init=ssm_init)
        elif variant == 'v2':
            self.mamba = nn.Sequential(
                Mamba(in_dim=mamba_input_dim, embed_dim=embed_dim, ssm_state="bidirectional", ssm_init=ssm_init),
                Mamba(in_dim=mamba_input_dim, embed_dim=embed_dim, ssm_state="forward", ssm_init=ssm_init)
            )
        elif variant == 'v3':
            self.mamba = nn.Sequential(
                Mamba(in_dim=mamba_input_dim, embed_dim=embed_dim, ssm_state="bidirectional", ssm_init=ssm_init),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU()
            )
        else:  # m3
            self.mamba = nn.ModuleList([
                Mamba(in_dim=mamba_input_dim, embed_dim=embed_dim, ssm_state="forward", ssm_init=ssm_init),
                Mamba(in_dim=mamba_input_dim, embed_dim=embed_dim, ssm_state="backward", ssm_init=ssm_init)
            ])

        self.post_proj = nn.Linear(mamba_input_dim, embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.pre_proj(x)

        if isinstance(self.mamba, nn.ModuleList):
            x_fw = self.mamba[0](x)
            x_bw = self.mamba[1](torch.flip(x, dims=[1]))
            x_bw = torch.flip(x_bw, dims=[1])
            x = x_fw + x_bw
        else:
            x = self.mamba(x)

        x = self.post_proj(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual
        return x

# Test the block
if __name__ == "__main__":
    B, H, W, C = 4, 32, 32, 60
    x = torch.randn(B, H * W, C)
    block = HDRMambaBlock(embed_dim=C, seq_len=H * W, variant='m3', ssm_init='s4d')
    out = block(x)
    print("Output shape:", out.shape)  # Should be (B, H*W, C)

    def center_crop_div32(x):
        C, H, W = x.shape
        H32 = (H // 32) * 32
        W32 = (W // 32) * 32
        start_h = (H - H32) // 2
        start_w = (W - W32) // 2
        return x[:, start_h:start_h + H32, start_w:start_w + W32]
    print(center_crop_div32(out).shape)