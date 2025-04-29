import torch
import torch.nn as nn
from torchvision import models

class VGG16_FeatureExtractor(nn.Module):
    def __init__(self, path_to_weights="vgg16.pth", requires_grad=False):
        super().__init__()
        vgg = models.vgg16()
        vgg.load_state_dict(torch.load(path_to_weights))

        # Extract relu1_2, relu2_2, relu3_3, relu4_3
        self.stage1 = nn.Sequential(*vgg.features[:4])    # relu1_2
        self.stage2 = nn.Sequential(*vgg.features[4:9])   # relu2_2
        self.stage3 = nn.Sequential(*vgg.features[9:16])  # relu3_3
        self.stage4 = nn.Sequential(*vgg.features[16:23]) # relu4_3

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        feats = []
        x = self.stage1(x); feats.append(x)
        x = self.stage2(x); feats.append(x)
        x = self.stage3(x); feats.append(x)
        x = self.stage4(x); feats.append(x)
        return feats

class LPIPSLoss(nn.Module):
    def __init__(self, vgg_weights_path="vgg16.pth"):
        super().__init__()
        self.vgg = VGG16_FeatureExtractor(path_to_weights=vgg_weights_path)
        self.l2 = nn.MSELoss()

    def forward(self, x, y):
        # Normalize from [0, 1] to [-1, 1] for VGG
        x = (x - 0.5) * 2
        y = (y - 0.5) * 2

        feats_x = self.vgg(x)
        feats_y = self.vgg(y)

        loss = 0
        for fx, fy in zip(feats_x, feats_y):
            loss += self.l2(fx, fy)
        return loss
