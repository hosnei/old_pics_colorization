#!/usr/bin/env python3
"""
Train U-Net for Image Colorization on Tiny ImageNet
"""

import os , sys
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir,'..'))
if project_root not in sys.path :
    sys.path.append(project_root)
from scripts.unet_colorization import UNet
from scripts.images64 import prepare_dataloaders
from torchmetrics.functional import structural_similarity_index_measure as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# Load data
train_loader, val_loader, test_loader = prepare_dataloaders()

# Model, loss, optimizer
model = UNet().to(device)
criterion = nn.L1Loss()                      # L1 loss (pixel-wise)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Training loop
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for gray, color in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        gray, color = gray.to(device), color.to(device)
        optimizer.zero_grad()
        preds = model(gray)
        loss = criterion(preds, color)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss, val_ssim = 0.0, 0.0
    with torch.no_grad():
        for gray, color in val_loader:
            gray, color = gray.to(device), color.to(device)
            preds = model(gray)
            val_loss += criterion(preds, color).item()
            val_ssim += ssim(preds, color).item()

    val_loss /= len(val_loader)
    val_ssim /= len(val_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train L1: {avg_train_loss:.4f} | "
          f"Val L1: {val_loss:.4f} | SSIM: {val_ssim:.4f}")

# Save model
torch.save(model.state_dict(), "results/checkpoints/unet_colorizer.pt")
print("✅ Model saved.")