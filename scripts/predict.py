#!/usr/bin/env python3
"""
Visualize U-Net colorization results
------------------------------------
Shows: Original color → Grayscale input → Model colorized output
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os, sys

# --- Ensure project root path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from scripts.unet_colorization import UNet
from scripts.tiny_image import prepare_dataloaders

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# --- Load data ---
_, _, test_loader = prepare_dataloaders(batch_size=8)

# --- Load model ---
model = UNet().to(device)
model.load_state_dict(torch.load("results/checkpoints/unet_colorizer.pt", map_location=device))
model.eval()

# --- Get a batch ---
gray, color = next(iter(test_loader))
gray, color = gray.to(device), color.to(device)

# --- Predict ---
with torch.no_grad():
    pred = model(gray)

# --- Convert tensors to numpy ---
def to_numpy(img_tensor):
    img_tensor = img_tensor.cpu().numpy().transpose(0, 2, 3, 1)  # [B,C,H,W] → [B,H,W,C]
    return np.clip(img_tensor, 0, 1)

gray_np = gray.cpu().numpy().transpose(0, 2, 3, 1)
pred_np = to_numpy(pred)
color_np = to_numpy(color)

# --- Plot results ---
n = 8  # how many samples to show
fig, axes = plt.subplots(3, n, figsize=(16, 6))

for i in range(n):
    # Original color (ground truth)
    axes[0, i].imshow(color_np[i])
    axes[0, i].axis('off')

    # Grayscale input
    axes[1, i].imshow(gray_np[i].squeeze(), cmap='gray')
    axes[1, i].axis('off')

    # Model prediction
    axes[2, i].imshow(pred_np[i])
    axes[2, i].axis('off')

axes[0, 0].set_ylabel("Original\nColor", fontsize=10)
axes[1, 0].set_ylabel("Gray\n(Input)", fontsize=10)
axes[2, 0].set_ylabel("Predicted\nColor", fontsize=10)

plt.suptitle("Tiny ImageNet Colorization: Original → Gray → Predicted", fontsize=14)
plt.tight_layout()
plt.show()