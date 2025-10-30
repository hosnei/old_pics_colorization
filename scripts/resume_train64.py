#!/usr/bin/env python3
"""
resume_train64.py — Continue training a pretrained U-Net colorizer
with lower learning rate, mixed loss (L1 + SSIM), and scheduler.
"""

import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir,'..'))
if project_root not in sys.path :
    sys.path.append(project_root)
from torchmetrics.functional import structural_similarity_index_measure as ssim
from scripts.train64 import UNet
from scripts.images64_split import get_dataloaders

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "results/checkpoints/unet_colorizer_10_color_loss.pt"
SAVE_DIR = "results/checkpoints"
EPOCHS = 10
LR = 1e-4  # Lower learning rate for fine-tuning

print(f"✅ Using device: {DEVICE}")

# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------
train_loader, val_loader, _ = get_dataloaders(batch_size=512,num_classes=10)

# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
print(f"✅ Loaded pretrained weights from {MODEL_PATH}")

# ---------------------------------------------------------------------
# OPTIMIZER & SCHEDULER
# ---------------------------------------------------------------------
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---------------------------------------------------------------------
# LOSS FUNCTION (L1 + SSIM)
# ---------------------------------------------------------------------
def color_loss(pred, target, alpha=0.5):
    l1 = torch.nn.functional.l1_loss(pred, target)
    s = 1 - ssim(pred, target)
    return alpha * l1 + (1 - alpha) * s

# ---------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------
best_val = float("inf")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0

    for gray, color in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        gray, color = gray.to(DEVICE), color.to(DEVICE)
        optimizer.zero_grad()
        pred = model(gray)
        loss = color_loss(pred, color)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    scheduler.step()

    # Validation
    model.eval()
    val_loss, val_ssim = 0.0, 0.0
    with torch.no_grad():
        for gray, color in val_loader:
            gray, color = gray.to(DEVICE), color.to(DEVICE)
            pred = model(gray)
            val_loss += color_loss(pred, color).item()
            val_ssim += ssim(pred, color).item()
    val_loss /= len(val_loader)
    val_ssim /= len(val_loader)

    print(f"Epoch [{epoch}/{EPOCHS}] "
          f"Train: {avg_train_loss:.4f} | "
          f"Val: {val_loss:.4f} | SSIM: {val_ssim:.4f}")
        # Save best model
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(),
               os.path.join(SAVE_DIR, f"unet_colorizer_10_color_loss{epoch}.pt"))
        print(f"💾 Saved model → unet_colorizer_10_color_loss{epoch}.pt")

print("✅ Fine-tuning complete!")