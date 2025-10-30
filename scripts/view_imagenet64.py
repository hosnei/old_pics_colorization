#!/usr/bin/env python3
"""
visualize_colorization.py
=========================
Loads your trained U-Net and visualizes grayscale → colorization results.
"""

import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from convert_imagenet64_to_images import get_dataloaders
from train_colorizer import UNet  # make sure train_colorizer.py is in the same folder

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
MODEL_PATH = "results/checkpoints/best_unet_colorizer.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 8  # number of images to show per row
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Loaded model from {MODEL_PATH}")

# ---------------------------------------------------------------------
# LOAD TEST DATA
# ---------------------------------------------------------------------
_, _, test_loader = get_dataloaders(batch_size=N_SAMPLES)
gray_batch, color_batch = next(iter(test_loader))
gray_batch, color_batch = gray_batch.to(DEVICE), color_batch.to(DEVICE)

# ---------------------------------------------------------------------
# GENERATE COLORIZED OUTPUT
# ---------------------------------------------------------------------
with torch.no_grad():
    preds = model(gray_batch).cpu()

# ---------------------------------------------------------------------
# CONVERT BACK TO [0,1] RANGE FOR DISPLAY
# ---------------------------------------------------------------------
def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)

gray_show = denormalize(gray_batch.cpu().repeat(1, 3, 1, 1))  # replicate 1→3 channels for display
color_show = denormalize(color_batch.cpu())
preds_show = denormalize(preds)

# ---------------------------------------------------------------------
# VISUALIZE
# ---------------------------------------------------------------------
def show_triplet(gray_imgs, color_imgs, pred_imgs, n=N_SAMPLES):
    """Display grayscale, colorized output, and ground-truth color."""
    fig, axes = plt.subplots(3, n, figsize=(2*n, 6))
    for i in range(n):
        axes[0, i].imshow(gray_imgs[i].permute(1, 2, 0))
        axes[1, i].imshow(pred_imgs[i].permute(1, 2, 0))
        axes[2, i].imshow(color_imgs[i].permute(1, 2, 0))
        for j in range(3):
            axes[j, i].axis("off")

    axes[0, 0].set_ylabel("Gray input", fontsize=12)
    axes[1, 0].set_ylabel("Predicted", fontsize=12)
    axes[2, 0].set_ylabel("Original", fontsize=12)
    plt.tight_layout()
    plt.show()

show_triplet(gray_show, color_show, preds_show, n=N_SAMPLES)