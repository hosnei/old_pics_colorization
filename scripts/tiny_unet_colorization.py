#!/usr/bin/env python3
"""
U-Net model for image colorization
----------------------------------
Input  : grayscale (1 x 64 x 64)
Output : color (3 x 64 x 64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

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
        self.down4 = UNetBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.middle = UNetBlock(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512,  kernel_size=2, stride=2)
        self.dec4 = UNetBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256,  kernel_size=2, stride=2)
        self.dec3 = UNetBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128,  kernel_size=2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2)
        self.dec1 = UNetBlock(128, 64)

        # Output (RGB)
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        # Bottleneck
        m = self.middle(p4)

        # Decoder + skip connections
        u4 = torch.cat([self.up4(m), d4], dim=1)
        u4 = self.dec4(u4)
        u3 = torch.cat([self.up3(u4), d3], dim=1)
        u3 = self.dec3(u3)
        u2 = torch.cat([self.up2(u3), d2], dim=1)
        u2 = self.dec2(u2)
        u1 = torch.cat([self.up1(u2), d1], dim=1)
        u1 = self.dec1(u1)

        return torch.sigmoid(self.final(u1))