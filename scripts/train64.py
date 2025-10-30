#!/usr/bin/env python3
"""
train_colorizer.py
==================
Train a U-Net on ImageNet-64 converted image folders for grayscale→color mapping.
"""

import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir,'..'))
if project_root not in sys.path :
    sys.path.append(project_root)
from images64_split import get_dataloaders

# ---------------------------------------------------------------------
# MODEL: Simple U-Net for colorization
# ---------------------------------------------------------------------
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.down1 = UNetBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = UNetBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = UNetBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = UNetBlock(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = UNetBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)

        # Output: 3 channels (RGB)
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        up3 = self.up3(b)
        merge3 = torch.cat([up3, d3], dim=1)
        dec3 = self.dec3(merge3)

        up2 = self.up2(dec3)
        merge2 = torch.cat([up2, d2], dim=1)
        dec2 = self.dec2(merge2)

        up1 = self.up1(dec2)
        merge1 = torch.cat([up1, d1], dim=1)
        dec1 = self.dec1(merge1)

        return self.final(dec1)



# LOSS FUNCTION (L1 + SSIM)
# ---------------------------------------------------------------------
def color_loss(pred, target, alpha=0.5):
    l1 = torch.nn.functional.l1_loss(pred, target)
    s = 1 - ssim(pred, target)
    return alpha * l1 + (1 - alpha) * s



# ---------------------------------------------------------------------
# TRAINING PIPELINE
# ---------------------------------------------------------------------
num_classes=10
def train_model(epochs=10, lr=1e-4, save_dir="results/checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Using device:", device)

    os.makedirs(save_dir, exist_ok=True)

    # Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(num_classes=10)

    # Model, loss, optimizer
    model = UNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        # ------------------ TRAIN ------------------
        model.train()
        running_loss = 0.0
        for gray, color in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            gray, color = gray.to(device), color.to(device)
            optimizer.zero_grad()
            preds = model(gray)
            loss = color_loss(preds, color)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train = running_loss / len(train_loader)

        # ------------------ VALIDATE ------------------
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

        print(f"Epoch [{epoch}/{epochs}] "
              f"Train L1: {avg_train:.4f} | "
              f"Val L1: {val_loss:.4f} | SSIM: {val_ssim:.4f}")

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f"unet_colorizer_{num_classes}_color_loss.pt"))
            print(f"💾 Saved best model at epoch {epoch}")

    print("✅ Training complete.")
    return model


# ---------------------------------------------------------------------
if __name__ == "__main__":
    train_model(epochs=10)