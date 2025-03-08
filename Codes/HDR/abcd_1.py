import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms.functional import rgb_to_grayscale

# --------------------------------------
# 1. Feature Extractor (Encoder)
# --------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feature_dim=64):
        super(FeatureExtractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)

# --------------------------------------
# 2. Pyramid Cross-Attention Alignment (PCA Module)
# --------------------------------------
class PyramidCrossAttention(nn.Module):
    def __init__(self, feature_dim=64):
        super(PyramidCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(feature_dim, feature_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(feature_dim, feature_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, ref, target):
        Q = self.query_conv(ref)
        K = self.key_conv(target)
        V = self.value_conv(target)
        
        attn = self.softmax(torch.matmul(Q.flatten(2), K.flatten(2).transpose(-2, -1)))
        aligned = torch.matmul(attn, V.flatten(2)).view_as(target)
        return aligned + ref

# --------------------------------------
# 3. Fusion Network
# --------------------------------------
class FusionNetwork(nn.Module):
    def __init__(self, feature_dim=64):
        super(FusionNetwork, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, 3, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.fusion(x)

# --------------------------------------
# 4. Generator (Full HDR Fusion Model)
# --------------------------------------
class HDRGenerator(nn.Module):
    def __init__(self, feature_dim=64):
        super(HDRGenerator, self).__init__()
        self.extractor = FeatureExtractor(in_channels=3, feature_dim=feature_dim)
        self.pca = PyramidCrossAttention(feature_dim=feature_dim)
        self.fusion_net = FusionNetwork(feature_dim=feature_dim)
    
    def forward(self, l1, l2, l3):
        f1, f2, f3 = self.extractor(l1), self.extractor(l2), self.extractor(l3)
        aligned_f1, aligned_f3 = self.pca(f2, f1), self.pca(f2, f3)
        fused_features = (aligned_f1 + f2 + aligned_f3) / 3.0
        return self.fusion_net(fused_features)

# --------------------------------------
# 5. Discriminator (WGAN-GP)
# --------------------------------------
class HDRDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(HDRDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
        )
    
    def forward(self, x):
        return self.model(x).view(-1)

# --------------------------------------
# 6. Loss Functions (WGAN-GP, Pixel, SSIM)
# --------------------------------------
def generator_loss(discriminator, fake_images):
    return -torch.mean(discriminator(fake_images))

def discriminator_loss(discriminator, real_images, fake_images, lambda_gp=10):
    real_preds = discriminator(real_images)
    fake_preds = discriminator(fake_images)
    d_loss = torch.mean(fake_preds) - torch.mean(real_preds)
    return d_loss + lambda_gp * gradient_penalty(discriminator, real_images, fake_images)

def gradient_penalty(discriminator, real_images, fake_images):
    alpha = torch.rand(real_images.size(0), 1, 1, 1).to(real_images.device)
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    interpolated.requires_grad_(True)
    d_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones_like(d_interpolated),
                                    create_graph=True, retain_graph=True)[0]
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def pixel_loss(fake, real):
    return F.mse_loss(fake, real)

def ssim_loss(fake, real):
    return 1 - F.l1_loss(rgb_to_grayscale(fake), rgb_to_grayscale(real))

# --------------------------------------
# 7. Training Loop
# --------------------------------------
# Define models
generator = HDRGenerator().cuda()
discriminator = HDRDiscriminator().cuda()

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

# Training Step
for epoch in range(100):
    for l1, l2, l3, gt in dataloader:
        l1, l2, l3, gt = l1.cuda(), l2.cuda(), l3.cuda(), gt.cuda()
        
        # Train Generator
        g_optimizer.zero_grad()
        fake_hdr = generator(l1, l2, l3)
        g_loss = generator_loss(discriminator, fake_hdr) + 0.1 * pixel_loss(fake_hdr, gt) + 0.1 * ssim_loss(fake_hdr, gt)
        g_loss.backward()
        g_optimizer.step()
        
        # Train Discriminator
        d_optimizer.zero_grad()
        d_loss = discriminator_loss(discriminator, gt, fake_hdr)
        d_loss.backward()
        d_optimizer.step()
        
    print(f"Epoch {epoch}: G Loss={g_loss.item()}, D Loss={d_loss.item()}")
