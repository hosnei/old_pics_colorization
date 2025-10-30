#!/usr/bin/env python3
"""
convert_imagenet64_to_images.py
===============================
Converts Downsampled ImageNet-64 .npz / pickle batches into normal PNG folders
with progress bars and ETA for large datasets.
"""

import os
import pickle
import numpy as np
import cv2
import random
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from multiprocessing import cpu_count

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
SRC_DIR = "/home/user/old_pics_colorization/data/imagenet64"  # path to raw pickle/npz batches
OUT_DIR = "/home/user/old_pics_colorization/data/converted64"    # output folder for PNGs
IMG_SIZE = 64
VAL_RATIO = 0.2
BATCH_SIZE = 128
SEED = 42
NUM_WORKERS = max(4, cpu_count() // 2)
# ---------------------------------------------------------------------


def load_batches(split="train"):
    """Load the raw numpy arrays from pickle/npz files."""
    if split == "train":
        files = sorted(f for f in os.listdir(SRC_DIR) if f.startswith("train_data_batch"))
    else:
        files = ["val_data"]

    data, labels = [], []
    for fname in tqdm(files, desc=f"Loading {split} batches", ncols=100):
        path = os.path.join(SRC_DIR, fname)
        with open(path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
        data.append(entry["data"])
        labels.extend(entry["labels"])

    data = np.concatenate(data).reshape(-1, 3, IMG_SIZE, IMG_SIZE)
    labels = np.array(labels) - 1
    return data, labels


def save_images(split, data, labels):
    """
    Save each image as PNG under OUT_DIR/split/class_id/.
    Displays a live progress bar and ETA.
    """
    print(f"\n🖼️  Saving {split} images to disk...")
    total = len(labels)

    for i, (img, label) in enumerate(tqdm(zip(data, labels), total=total, ncols=100, smoothing=0.1)):
        img = np.transpose(img, (1, 2, 0)).astype(np.uint8)  # (H,W,C)
        class_dir = os.path.join(OUT_DIR, split, f"{label:04d}")
        os.makedirs(class_dir, exist_ok=True)
        path = os.path.join(class_dir, f"{i:06d}.png")
        # BGR for cv2
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"✅ {split} export complete → {os.path.join(OUT_DIR, split)}")


def convert_all():
    """Convert train/val/test sets to PNG folders."""
    random.seed(SEED)
    np.random.seed(SEED)

    print("\n🚀 Starting ImageNet64 conversion → PNG folders")
    train_data, train_labels = load_batches("train")

    # Split train/val
    n = len(train_data)
    idx = np.arange(n)
    np.random.shuffle(idx)
    val_size = int(VAL_RATIO * n)
    val_idx, train_idx = idx[:val_size], idx[val_size:]

    save_images("train", train_data[train_idx], train_labels[train_idx])
    save_images("val", train_data[val_idx], train_labels[val_idx])

    # Test set (original val_data)
    test_data, test_labels = load_batches("val")
    save_images("test", test_data, test_labels)

    print(f"\n🎯 Conversion complete! All images saved under: {OUT_DIR}")


# ---------------------------------------------------------------------
# DATASET LOADER
# ---------------------------------------------------------------------
def get_dataloaders(root=OUT_DIR, batch_size=BATCH_SIZE, num_classes=None, seed=SEED):
    """
    Return train/val/test dataloaders yielding (gray, color) pairs.
    Optionally restricts to a subset of num_classes (e.g. 100 out of 1000).
    """

    random.seed(seed)
    np.random.seed(seed)

    # Create folder to store chosen subset
    os.makedirs("results", exist_ok=True)
    subset_file = os.path.join("results", f"chosen_classes_{num_classes}.txt")

    transform_color = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_gray = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        imgs, labels = zip(*batch)
        color_tensors = torch.stack([transform_color(img) for img in imgs])
        gray_tensors = torch.stack([transform_gray(img) for img in imgs])
        return gray_tensors, color_tensors
    
    def get_chosen_classes(all_classes):
        if num_classes is None or num_classes >= len(all_classes):
            return all_classes  # use full set

        if os.path.exists(subset_file):
            with open(subset_file, "r") as f:
                chosen = [line.strip() for line in f.readlines()]
            print(f"📂 Loaded existing class subset from {subset_file} ({len(chosen)} classes)")
        else:
            chosen = sorted(random.sample(all_classes, num_classes))
            with open(subset_file, "w") as f:
                f.write("\n".join(chosen))
            print(f"📝 Saved new subset of {num_classes} classes → {subset_file}")
        return chosen

    def make_loader(split, shuffle=False):
        dataset = ImageFolder(os.path.join(root, split))

        # ------------------------------------------------------------------
        # Subsample classes if requested
        # ------------------------------------------------------------------
        if num_classes is not None:
            all_classes = sorted(dataset.classes)
            chosen = get_chosen_classes(all_classes)
            print(f"📦 Using {num_classes} classes for {split} set")

            # Filter dataset samples to keep only chosen classes
            filtered_samples = [
                (path, label) for path, label in dataset.samples
                if dataset.classes[label] in chosen
            ]
            dataset.samples = filtered_samples
            dataset.imgs = filtered_samples  # compatibility for torchvision
            dataset.targets = [s[1] for s in filtered_samples]

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_fn
        )
        return loader

    # ------------------------------------------------------------------
    # Build all dataloaders
    # ------------------------------------------------------------------
    train_loader = make_loader("train", shuffle=True)
    val_loader   = make_loader("val")
    test_loader  = make_loader("test")

    print(f"✅ Dataloaders ready | train={len(train_loader.dataset)} | val={len(val_loader.dataset)} | test={len(test_loader.dataset)}")
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
   
    convert_all()  # one-time conversion

    # Quick sanity check
    train_loader, val_loader, test_loader = get_dataloaders()
    gray, color = next(iter(train_loader))
    print("Gray:", gray.shape, "Color:", color.shape)