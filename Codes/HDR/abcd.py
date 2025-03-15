import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms.functional import rgb_to_grayscale

# --------------------------------------
# 1. Feature Extractor (Attention Pyramid Transformer)
# --------------------------------------
class AttentionPyramidTransformer(nn.Module):
    def __init__(self, in_channels, n_feats, num_heads=4, kernel_sizes=[3,3,3], strides=[1,2,2], paddings=[1,1,1]):
        super(AttentionPyramidTransformer, self).__init__()
        self.in_channels = in_channels
        self.n_feats = n_feats
        self.num_heads = num_heads
        
        layers = []
        in_channel = self.in_channels       
        for i in range(len(kernel_sizes)):
            cur_layer = nn.Sequential(
                nn.Conv2d(in_channel, self.n_feats, kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i]),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.MultiheadAttention(embed_dim=self.n_feats, num_heads=self.num_heads, batch_first=True)
            )
            layers.append(cur_layer)
            in_channel = self.n_feats
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        feature_list = []
        for layer in self.layers:
            x = layer[0](x)  # Convolution
            b, c, h, w = x.shape
            x = x.view(b, c, -1).permute(0, 2, 1)  # Reshape for MultiheadAttention
            x, _ = layer[1](x, x, x)  # Self-Attention
            x = x.permute(0, 2, 1).view(b, c, h, w)  # Reshape back to feature map
            feature_list.append(x)
        return feature_list

# --------------------------------------
# 2. Multi-Cross Attention Alignment with HDR Fusion
# --------------------------------------
class HDRFusionGenerator(nn.Module):
    def __init__(self, in_c, num_heads=4, dim_align=64):
        super(HDRFusionGenerator, self).__init__()
        
        # Shallow feature extraction
        self.conv_f1 = nn.Conv2d(in_c, dim_align, 3, 1, 1)
        self.conv_f2 = nn.Conv2d(in_c, dim_align, 3, 1, 1)
        self.conv_f3 = nn.Conv2d(in_c, dim_align, 3, 1, 1)

        # Multi-Scale Feature Extraction with Attention Pyramid Transformer
        self.pyramid1 = AttentionPyramidTransformer(in_channels=dim_align, n_feats=dim_align, num_heads=num_heads)
        self.pyramid2 = AttentionPyramidTransformer(in_channels=dim_align, n_feats=dim_align, num_heads=num_heads)
        self.pyramid3 = AttentionPyramidTransformer(in_channels=dim_align, n_feats=dim_align, num_heads=num_heads)

        # HDR Fusion Network
        self.fusion = nn.Sequential(
            nn.Conv2d(dim_align * 3, dim_align, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim_align, in_c, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x1, x2, x3):
        x1, x2, x3 = self.conv_f1(x1), self.conv_f2(x2), self.conv_f3(x3)
        
        x1_feature_list = self.pyramid1(x1)
        x2_feature_list = self.pyramid2(x2)
        x3_feature_list = self.pyramid3(x3)
        
        # HDR Fusion
        fused_features = torch.cat([x1_feature_list[-1], x2_feature_list[-1], x3_feature_list[-1]], dim=1)
        hdr_output = self.fusion(fused_features)
        
        return hdr_output

# --------------------------------------
# 3. Loss Functions
# --------------------------------------
def pixel_loss(fake, real):
    return F.mse_loss(fake, real)

def ssim_loss(fake, real):
    return 1 - F.l1_loss(rgb_to_grayscale(fake), rgb_to_grayscale(real))

# --------------------------------------
# 4. Training Loop
# --------------------------------------
def train_model(generator, dataloader, epochs=10, lr=1e-4, device='cuda'):
    optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    generator.train()
    
    for epoch in range(epochs):
        for l1, l2, l3, gt in dataloader:
            l1, l2, l3, gt = l1.to(device), l2.to(device), l3.to(device), gt.to(device)
            
            optimizer.zero_grad()
            fake_hdr = generator(l1, l2, l3)
            loss = pixel_loss(fake_hdr, gt) + 0.1 * ssim_loss(fake_hdr, gt)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# --------------------------------------
# 5. Random Input-Output Verification Test
# --------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HDRFusionGenerator(in_c=3).to(device)
    
    # Generate random multi-exposure images
    l1 = torch.rand(1, 3, 128, 128).to(device)  # Under-exposed image
    l2 = torch.rand(1, 3, 128, 128).to(device)  # Mid-exposure image
    l3 = torch.rand(1, 3, 128, 128).to(device)  # Over-exposed image
    gt = torch.rand(1, 3, 128, 128).to(device)  # Ground truth HDR image
    
    # Run through model
    output = model(l1, l2, l3)
    print("Output shape:", output.shape)
    
    # Run training with random data
    print("Starting training loop...")
    random_dataloader = [(l1, l2, l3, gt) for _ in range(5)]  # Simulated dataset
    train_model(model, random_dataloader, epochs=5, device=device)
