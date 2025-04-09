import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class HDRLoss(nn.Module):
    def __init__(self, mu=5000, lambda_p=0.01):
        super(HDRLoss, self).__init__()
        self.mu = mu
        self.lambda_p = lambda_p
        
        # Load VGG16 up to reluX layers for perceptual loss
        vgg = models.vgg16(pretrained=True).features
        self.vgg_layers = nn.Sequential(*list(vgg)[:16]).eval()  # You can change the number of layers
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def tonemap(self, x):
        """Apply µ-law tonemapping."""
        return torch.log(1 + self.mu * x) / torch.log(torch.tensor(1 + self.mu, dtype=x.dtype, device=x.device))

    def perceptual_features(self, x):
        """Extract features from VGG for perceptual loss."""
        return self.vgg_layers(x)

    def forward(self, pred, target):
        # Tonemap predicted and target HDR images
        pred_tm = self.tonemap(pred)
        target_tm = self.tonemap(target)

        # l1 loss between tonemapped pred and target
        l1_loss = F.l1_loss(pred_tm, target_tm)

        # Perceptual loss between tonemapped pred and target
        pred_features = self.perceptual_features(pred_tm)
        target_features = self.perceptual_features(target_tm)
        perceptual_loss = F.l1_loss(pred_features, target_features)

        # Total loss
        total_loss = l1_loss + self.lambda_p * perceptual_loss
        return total_loss
