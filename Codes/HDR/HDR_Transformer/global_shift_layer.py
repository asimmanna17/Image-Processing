import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalShiftLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dx = nn.Parameter(torch.tensor(0.0))
        self.dy = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        B, C, H, W = x.shape
        theta = torch.tensor([
            [1, 0, -self.dx * 2 / W],
            [0, 1, -self.dy * 2 / H]
        ], dtype=x.dtype, device=x.device).unsqueeze(0).repeat(B, 1, 1)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        shifted_x = F.grid_sample(x, grid, padding_mode='border', align_corners=True)
        return shifted_x
# Assuming these are defined elsewhere
alignment_net = YourAlignmentNetwork()
hdr_net = YourHDRNetwork()
global_shift = GlobalShiftLayer()  # << Add this

# Optimizer with all parameters including dx/dy
optimizer = torch.optim.Adam(
    list(alignment_net.parameters()) + 
    list(hdr_net.parameters()) + 
    list(global_shift.parameters()), 
    lr=1e-4
)
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # raw input frames (B, C, H, W)
    aligned_img = alignment_net(input1, input2, input3)  # outputs aligned to input2

    # Apply learnable global shift
    aligned_img_shifted = global_shift(aligned_img)

    # Predict HDR
    pred_hdr = hdr_net(aligned_img_shifted)

    # Compressed L1 Loss with GT HDR
    loss = compressed_l1_loss(pred_hdr, gt_hdr)  # your custom loss

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}: Loss={loss.item():.4f}, Shift dx={global_shift.dx.item():.2f}, dy={global_shift.dy.item():.2f}")
