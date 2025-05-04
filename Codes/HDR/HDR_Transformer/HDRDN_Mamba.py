import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from alignhead import *
from HDR_transformer import *
from mamba import HDRMambaBlock

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

class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super(SingleMambaBlock, self).__init__()
        self.encoder = HDRMambaBlock(dim, seq_len=dim, variant=None)
        self.norm = LayerNorm(dim,'with_bias')
        # self.PatchEmbe=PatchEmbed(patch_size=4, stride=4,in_chans=dim, embed_dim=dim*16)
    def forward(self,ipt):
        x,residual = ipt
        residual = x+residual
        x = self.norm(residual)
        return (self.encoder(x),residual)
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight 
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
     
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape)==4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops

class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops

class HDRDN_Mamba(nn.Module):
    def __init__(self, n_channel =8, out_channel = 4, embed_dim = 60, depths=[4, 4, 4]):
        super(HDRDN_Mamba, self).__init__()
        self.in_channel = n_channel
        self.out_channel = out_channel
        embed_dim = embed_dim
        self.embed_dim = embed_dim

        img_size=128
        patch_size=1
        norm_layer=nn.LayerNorm
        patch_norm=True
        self.patch_norm = patch_norm
        
        #################### 1. Pyramid Cross-Attention Alignment Module #############################################
        self.align_head = MultiCrossAlign_head_atttrans_res1sepalign(in_c=self.in_channel, dim_align=self.embed_dim)
        #################### 2. Spatial Attention Module #############################################################
        self.att1 = SpatialAttentionModule(self.embed_dim)
        self.att2 = SpatialAttentionModule(self.embed_dim)
        self.conv_first = nn.Conv2d(self.embed_dim *3, self.embed_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #####################################################################################################
        ################################ 5, fused image reconstruction ################################
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=224, patch_size=4, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
    
        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.feature_re = nn.Sequential(*[SingleMambaBlock(self.embed_dim) for i in range(1)])
        self.conv_last1 = nn.Conv2d(embed_dim, int(self.embed_dim/2), 3, 1, 1)
        self.conv_last2 = nn.Conv2d(int(self.embed_dim/2), int(self.embed_dim/4), 3, 1, 1)
        self.conv_last3 = nn.Conv2d(int(self.embed_dim/4), self.out_channel, 3, 1, 1)

        
        self.apply(self._init_weights)

    def forward(self, x1, x2, x3):
        f1_att, f2, f3_att, f1, f3 = self.align_head(x1, x2, x3)
        f1_att = f2 + f1_att
        f3_att = f2 + f3_att

        f1_att = self.att1(f1_att, f2)*f1_att
        f3_att = self.att2(f3_att, f2)*f3_att

        x = self.conv_first(torch.cat((f1_att, f2, f3_att), axis=1))

        x = self.fused_img_recon(x)

        
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

    def fused_img_recon(self, x):        
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        #print(x.shape)
        # -------------------mamba------------------ #
        residual_re = 0
        x,residual_re = self.feature_re([x,residual_re])
  
        x = self.patch_unembed(x, x_size)
        
        # -------------------Convolution------------------- #
        x = self.lrelu(self.conv_last1(x))
        x = self.lrelu(self.conv_last2(x))
        x = self.conv_last3(x) 
        return x


# Test the block
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_model = HDRDN_Mamba(n_channel =8, out_channel = 4, embed_dim = 60, depths=[4, 4, 4])
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
    print("Output shape:", generate_hdr.shape)  # Should be (B, H*W, C)