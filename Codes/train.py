import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from einops import rearrange

# Dataset Loader for Mobile-HDR
class MobileHDRDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        scenes = sorted(os.listdir(self.data_dir))
        for scene in scenes:
            scene_path = os.path.join(self.data_dir, scene)
            if os.path.isdir(scene_path):
                ldr_images = sorted([os.path.join(scene_path, f) for f in os.listdir(scene_path) if f.endswith('.npy')])
                hdr_image = os.path.join(scene_path, 'HDR.npy')
                if len(ldr_images) == 3 and os.path.exists(hdr_image):
                    samples.append((ldr_images, hdr_image))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ldr_paths, hdr_path = self.samples[idx]
        ldr_images = [np.load(path) for path in ldr_paths]
        hdr_image = np.load(hdr_path)

        # Normalize and Convert to Torch Tensors
        ldr_images = [torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0 for img in ldr_images]
        hdr_image = torch.tensor(hdr_image, dtype=torch.float32).unsqueeze(0) / 255.0
        ldr_stack = torch.stack(ldr_images, dim=0)  # Shape: (3, H, W)

        if self.transform:
            ldr_stack = self.transform(ldr_stack)
            hdr_image = self.transform(hdr_image)
        
        return ldr_stack, hdr_image

# Loss Functions
class HDRLoss(nn.Module):
    def __init__(self, mu=5000):
        super().__init__()
        self.mu = mu

    def forward(self, pred, target):
        pred_tone = torch.log(1 + self.mu * pred) / torch.log(1 + self.mu)
        target_tone = torch.log(1 + self.mu * target) / torch.log(1 + self.mu)
        return F.l1_loss(pred_tone, target_tone)

class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return 1 - torch.mean(torch.nn.functional.smooth_l1_loss(pred, target))

# Training Function
def train(model, dataloader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for ldr_stack, hdr_gt in dataloader:
            optimizer.zero_grad()
            hdr_pred = model(ldr_stack)
            loss = criterion(hdr_pred, hdr_gt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.6f}")

# Evaluation Metrics
def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))

# Instantiate Model and Dataset
model = HDRPipeline()
dataset = MobileHDRDataset(data_dir='/path/to/Mobile-HDR')
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training Setup
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
criterion = HDRLoss()

# Train Model
train(model, dataloader, optimizer, criterion, num_epochs=10)

print("Training complete!")
