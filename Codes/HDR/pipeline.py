import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import cv2
import os
glob

# ----------- DataLoader to load three exposure RAW images -----------
class HDRDataset(data.Dataset):
    def __init__(self, raw_dir, transform=None):
        self.raw_dir = raw_dir
        self.image_paths = sorted(glob.glob(os.path.join(raw_dir, "*.raw")))
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths) // 3  # Three exposures per HDR sample
    
    def __getitem__(self, index):
        paths = self.image_paths[index * 3:(index + 1) * 3]
        images = [np.fromfile(path, dtype=np.uint16).reshape(512, 512) / 65535.0 for path in paths]  # Normalize
        images = np.stack(images, axis=0)  # Shape (3, H, W)
        if self.transform:
            images = self.transform(images)
        return torch.tensor(images, dtype=torch.float32)  # Shape (3, H, W)

# ----------- Network: Hybrid Transformer + Spatial Filter Prediction -----------
class SpatialAdaptiveFilter(nn.Module):
    def __init__(self, channels):
        super(SpatialAdaptiveFilter, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 3 * 3, kernel_size=3, padding=1)  # 3x3 dynamic filter for each pixel
        
    def forward(self, x):
        filters = self.conv1(x)
        filters = self.conv2(filters)
        return filters.view(x.shape[0], 3, 3, x.shape[2], x.shape[3])  # (B, 3, 3, H, W)

class HDRFusionNet(nn.Module):
    def __init__(self):
        super(HDRFusionNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.transformer = nn.TransformerEncoderLayer(d_model=64, nhead=4)
        self.spatial_filter = SpatialAdaptiveFilter(64)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        encoded = self.transformer(encoded.flatten(2).transpose(1, 2)).transpose(1, 2).view_as(encoded)
        filters = self.spatial_filter(encoded)  # Predict dynamic filters
        fused = torch.zeros_like(x[:, :1])  # Initialize HDR output
        for i in range(3):
            fused += nn.functional.conv2d(x[:, i:i+1], filters[:, i], padding=1)
        hdr_output = self.decoder(fused)
        return hdr_output

# ----------- Loss Function: GIF Loss + Tone-Mapped L1 Loss -----------
def gradient_map(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(sobelx**2 + sobely**2)

def gif_loss(hdr_pred, exposures):
    grad_pred = gradient_map(hdr_pred.cpu().detach().numpy())
    grad_exposures = [gradient_map(exp.cpu().detach().numpy()) for exp in exposures]
    grad_max = np.maximum.reduce(grad_exposures)
    return torch.mean(torch.abs(torch.tensor(grad_pred) - torch.tensor(grad_max)))

def tone_mapped_loss(hdr_pred, hdr_gt, mu=5000):
    T = lambda x: torch.log(1 + mu * x) / torch.log(torch.tensor(1 + mu, dtype=torch.float32))
    return nn.L1Loss()(T(hdr_pred), T(hdr_gt))

# ----------- Training Setup -----------
dataset = HDRDataset("/path/to/raw_dataset")
dataloader = data.DataLoader(dataset, batch_size=4, shuffle=True)
model = HDRFusionNet().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train_one_epoch():
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = batch.cuda()
        optimizer.zero_grad()
        hdr_pred = model(batch)
        loss = gif_loss(hdr_pred, batch) + tone_mapped_loss(hdr_pred, batch.mean(dim=1, keepdim=True))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss: {total_loss / len(dataloader):.4f}")

# ----------- Inference & Save HDR Output -----------
def save_hdr_image(model, raw_images):
    model.eval()
    with torch.no_grad():
        hdr_output = model(torch.tensor(raw_images, dtype=torch.float32).unsqueeze(0).cuda())
    hdr_output = hdr_output.squeeze().cpu().numpy()
    cv2.imwrite("output.hdr", hdr_output * 255)

# Train and save HDR
for epoch in range(20):
    train_one_epoch()
save_hdr_image(model, dataset[0])

