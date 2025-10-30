#!/usr/bin/env python3
"""
Tiny ImageNet Setup & DataLoader Builder
========================================
- Downloads and extracts Tiny ImageNet (64x64)
- Fixes the validation folder structure
- Prepares PyTorch DataLoaders for training, validation, and testing
- Ideal for CNN / U-Net colorization projects
"""

import os
import zipfile
import urllib.request
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "data"
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
EXTRACTED_DIR = os.path.join(DATA_DIR, "tiny-imagenet-200")
BATCH_SIZE = 64
IMG_SIZE = 64
VAL_SPLIT_RATIO = 0.2   # 20% of train for validation
TEST_SPLIT_RATIO = 0.5  # 50% of original val for testing

# ============================================================
# DOWNLOAD + EXTRACT
# ============================================================
def download_tiny_imagenet():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "tiny-imagenet-200.zip")

    if not os.path.exists(EXTRACTED_DIR):
        if not os.path.exists(zip_path):
            print("📥 Downloading Tiny ImageNet (≈250 MB)...")
            with urllib.request.urlopen(TINY_IMAGENET_URL) as response, open(zip_path, 'wb') as out_file:
                total_length = int(response.getheader('content-length'))
                with tqdm(total=total_length, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in iter(lambda: response.read(1024 * 1024), b''):
                        out_file.write(chunk)
                        pbar.update(len(chunk))
        print("📦 Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("✅ Extraction complete.")
    else:
        print("✅ Tiny ImageNet already extracted.")

# ============================================================
# FIX VALIDATION FOLDER STRUCTURE
# ============================================================
def fix_validation_folder():
    val_dir = os.path.join(EXTRACTED_DIR, "val")
    val_img_dir = os.path.join(val_dir, "images")
    val_annot = os.path.join(val_dir, "val_annotations.txt")

    if not os.path.exists(val_img_dir):
        return

    print("🛠️ Fixing Tiny ImageNet validation structure...")
    with open(val_annot, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            img_name, class_name = parts[0], parts[1]
            class_dir = os.path.join(val_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            src = os.path.join(val_img_dir, img_name)
            dst = os.path.join(class_dir, img_name)
            if os.path.exists(src):
                os.rename(src, dst)

    if os.path.exists(val_img_dir):
        os.rmdir(val_img_dir)
    print("✅ Validation folder fixed.")

# ============================================================
# DATASET HELPERS
# ============================================================
def prepare_dataloaders(batch_size=BATCH_SIZE, image_size=IMG_SIZE):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    train_path = os.path.join(EXTRACTED_DIR, "train")
    val_path   = os.path.join(EXTRACTED_DIR, "val")

    # --- Load datasets ---
    train_data_full = datasets.ImageFolder(train_path, transform=transform)
    val_data_full   = datasets.ImageFolder(val_path, transform=transform)

    # --- Split train into train/val ---
    val_size = int(VAL_SPLIT_RATIO * len(train_data_full))
    train_size = len(train_data_full) - val_size
    train_data, val_data = random_split(train_data_full, [train_size, val_size])

    # --- Split val into val/test (equal halves) ---
    test_size = int(TEST_SPLIT_RATIO * len(val_data_full))
    val_size_new = len(val_data_full) - test_size
    val_data_full, test_data = random_split(val_data_full, [val_size_new, test_size])

    # --- Wrap datasets for colorization ---
    train_data = ColorizationDataset(train_data)
    val_data   = ColorizationDataset(val_data)
    test_data  = ColorizationDataset(test_data)

    # --- DataLoaders ---
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"✅ Dataset ready:")
    print(f"   • Train images: {len(train_data)}")
    print(f"   • Val images:   {len(val_data)} (from train)")
    print(f"   • Test images:  {len(test_data)} (from original val)")
    print(f"   • Classes:      {len(train_data_full.classes)}")

    return train_loader, val_loader, test_loader

# ============================================================
# COLORIZATION DATASET WRAPPER
# ============================================================
class ColorizationDataset:
    """
    Wraps an existing ImageFolder dataset and returns
    (grayscale_input, color_target) pairs.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]     # ignore class label
        # img is RGB tensor (3xHxW)
        gray = transforms.functional.rgb_to_grayscale(img, num_output_channels=1)
        return gray, img

# ============================================================
# SANITY CHECK VISUALIZATION
# ============================================================
def show_sample_batch(loader):
    imgs, labels = next(iter(loader))
    grid_img = np.transpose(imgs[:8], (0, 2, 3, 1))
    fig, axes = plt.subplots(1, 8, figsize=(16, 2))
    for i in range(8):
        axes[i].imshow(grid_img[i])
        axes[i].axis('off')
    plt.suptitle("Tiny ImageNet Sample Batch (64x64)")
    plt.show()
def show_gray_vs_color(loader):
    gray, color = next(iter(loader))
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(gray[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(np.transpose(color[i], (1, 2, 0)))
        axes[1, i].axis('off')
    plt.suptitle("Top: Grayscale inputs  |  Bottom: Color targets")
    plt.show()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    download_tiny_imagenet()
    fix_validation_folder()
    train_loader, val_loader, test_loader = prepare_dataloaders()

    # Show quick sample
    show_sample_batch(train_loader)
    show_gray_vs_color(train_loader)