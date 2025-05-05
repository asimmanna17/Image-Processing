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
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.conv1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(dim * 2)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(dim)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        self.att_map = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x1, x2):
        f_cat = torch.cat((x1, x2), dim=1)  # [B, 2C, H, W]
        x = self.conv1(f_cat)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        x = self.att_map(x)
        x = self.sigmoid(x)

        x = self.dropout(x)
        out = x1 * x + x1  # Residual attention
        return out



class HybridAttentionModule(nn.Module):
    def __init__(self, dim, reduction=8, dropout=0.1):
        super().__init__()
        self.dim = dim

        # ----- Spatial Attention Path -----
        self.s_conv1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, bias=False)
        self.s_bn1 = nn.BatchNorm2d(dim * 2)
        self.s_act1 = nn.LeakyReLU(0.2, inplace=True)

        self.s_conv2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False)
        self.s_bn2 = nn.BatchNorm2d(dim)
        self.s_act2 = nn.LeakyReLU(0.2, inplace=True)

        self.s_map = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.s_sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout2d(p=dropout)

        # ----- Channel Attention (SE Block) -----
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(dim, dim // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim // reduction, dim, bias=False)
        self.c_sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # ------ Spatial Attention ------
        f_cat = torch.cat((x1, x2), dim=1)
        s = self.s_act1(self.s_bn1(self.s_conv1(f_cat)))
        s = self.s_act2(self.s_bn2(self.s_conv2(s)))
        s = self.s_sigmoid(self.s_map(s))
        s = self.dropout(s)
        x_spatial = x1 * s  # Spatially modulated x1

        # ------ Channel Attention ------
        b, c, _, _ = x_spatial.shape
        ch = self.avg_pool(x_spatial).view(b, c)
        ch = self.relu(self.fc1(ch))
        ch = self.c_sigmoid(self.fc2(ch)).view(b, c, 1, 1)
        x_channel = x_spatial * ch  # Channel modulated

        # ------ Residual Add ------
        return x_channel + x1


class Joint_HDRDN(nn.Module):
    def __init__(self, n_channel =8, out_channel = 4, embed_dim = 60, depths=[4, 4, 4]):
        super(Joint_HDRDN, self).__init__()
        self.in_channel = n_channel
        self.out_channel = out_channel
        embed_dim = embed_dim
        self.embed_dim = embed_dim
        #################### 1. Pyramid Cross-Attention Alignment Module #############################################
        self.align_head = MultiCrossAlign_head_atttrans_res1sepalign(in_c=self.in_channel, dim_align=self.embed_dim)
        #################### 2. Spatial Attention Module #############################################################
        self.att1 = HybridAttentionModule(self.embed_dim)
        self.att2 = HybridAttentionModule(self.embed_dim)
        self.conv_first = nn.Conv2d(self.embed_dim *3, self.embed_dim, 3, 1, 1)
        #################### 3. Context-aware Swin Transformer blocks (CTBs)
        depths = depths
        num_heads = [6, 6, 6]
        img_size=128
        patch_size=1
        window_size=8 
        mlp_ratio=4. 
        qkv_bias=True
        qk_scale=None
        drop_rate=0.
        attn_drop_rate=0.
        drop_path_rate=0.1
        norm_layer=nn.LayerNorm
        resi_connection='1conv'
        ape=False
        patch_norm=True
        use_checkpoint=False
        self.num_layers = len(depths)
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros([1, num_patches, embed_dim]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(drop_rate)

        # stochastic depth
        dpr = [x.item() for x in np.linspace(0, drop_path_rate, sum(depths))] 

        # build Context-aware Swin Transformer blocks (CTBs)
        self.layers = []
        for i_layer in range(self.num_layers):
            layer = ContextAwareTransformerBlock(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)
        self.norm = norm_layer(self.num_features)
        
        # build the last conv layer
        self.conv_after_body = nn.Conv2d(self.embed_dim, self.embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(self.embed_dim, self.out_channel, 3, 1, 1)
        self.act_last = nn.Sigmoid()
        self.apply(self._init_weights)

    def forward(self, x1, x2, x3):
        f1_att, f2, f3_att, f1, f3 = self.align_head(x1, x2, x3)
        f1_att = f2 + f1_att
        f3_att = f2 + f3_att

        f1_att = self.att1(f1_att, f2)*f1_att
        f3_att = self.att2(f3_att, f2)*f3_att

        x = self.conv_first(torch.cat((f1_att, f2, f3_att), axis=1))

        # CTBs for HDR reconstruction
        res = self.conv_after_body(self.forward_features(x) + x)
        x = self.conv_last(f2 + res)
        
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



# Test the block
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_model = Joint_HDRDN(n_channel =8, out_channel = 4, embed_dim = 60, depths=[4, 4, 4])
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