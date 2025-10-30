#!/usr/bin/env python3
"""
visualize_by_class.py
=====================
Selects a specific class from the test set and visualizes colorization results
for 6 random images: (gray input, predicted color, original color).
"""

import os, sys
import torch
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir,'..'))
if project_root not in sys.path :
    sys.path.append(project_root)
from train64 import UNet
from images64_split import OUT_DIR, IMG_SIZE

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
MODEL_PATH = "results/checkpoints/unet_colorizer_1k.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 7
CUSTOM_DIR ="data/online_test_images/test2"
GRAY_TEMP_DIR = "data/online_test_images/gray_version"
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------
def denormalize(tensor):
    """Convert from [-1,1] → [0,1] for display."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)

def load_image(path):
    """Load an image from disk and return tensor pairs (gray, color)."""
    to_color = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    to_gray = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = Image.open(path).convert("RGB")
    return to_gray(img).unsqueeze(0), to_color(img).unsqueeze(0)


# AUTO-GRAYSCALE REAL IMAGES
# ---------------------------------------------------------------------
def create_grayscale_versions(src_folder=CUSTOM_DIR, dst_folder=GRAY_TEMP_DIR):
    """Converts all color images in src_folder to grayscale and saves to dst_folder."""
    if not os.path.exists(src_folder):
        print(f"❌ Folder not found: {src_folder}")
        return False

    os.makedirs(dst_folder, exist_ok=True)
    imgs = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(imgs) == 0:
        print(f"⚠️ No images found in {src_folder}")
        return False

    for img_name in imgs:
        path = os.path.join(src_folder, img_name)
        gray_path = os.path.join(dst_folder, img_name)
        try:
            img = Image.open(path).convert("L")  # convert to grayscale
            img.save(gray_path)
        except Exception as e:
            print(f"⚠️ Skipped {img_name}: {e}")
    print(f"✅ Created grayscale copies in {dst_folder}")
    return True

def visualize_samples(gray_imgs, preds, color_imgs, title):
    """Display triplets: grayscale, predicted (enhanced), original."""
    gray_show = denormalize(gray_imgs.repeat(1, 3, 1, 1))
    color_show = denormalize(color_imgs)
    preds_show = denormalize(preds)
   
   

    n = gray_show.size(0)
    fig, axes = plt.subplots(3, n, figsize=(2*n, 8))

    for i in range(n):

        axes[0, i].imshow(gray_show[i].permute(1, 2, 0))
        axes[1, i].imshow(preds_show[i].permute(1, 2, 0))
        axes[2, i].imshow(color_show[i].permute(1, 2, 0))
        
       
        for j in range(3):
            axes[j, i].axis("off")

    axes[0, 0].set_ylabel("Gray input", fontsize=12)
    axes[1, 0].set_ylabel("Predicted + enhanced", fontsize=12)
    axes[2, 0].set_ylabel("Original", fontsize=12)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def visualize_class(model, class_id, n=NUM_SAMPLES):
    """Visualize samples from ImageNet test subset."""
    class_dir = os.path.join(OUT_DIR, "test", f"{class_id:04d}")
    if not os.path.exists(class_dir):
        print(f"❌ Class {class_id} not found at {class_dir}")
        return

    imgs = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    chosen_imgs = random.sample(imgs, min(n, len(imgs)))

    gray_imgs, color_imgs, preds = [], [], []

    for img_name in chosen_imgs:
        gray_tensor, color_tensor = load_image(os.path.join(class_dir, img_name))
        gray_tensor, color_tensor = gray_tensor.to(DEVICE), color_tensor.to(DEVICE)
        with torch.no_grad():
            pred = model(gray_tensor).cpu()
        gray_imgs.append(gray_tensor.cpu())
        color_imgs.append(color_tensor.cpu())
        preds.append(pred)

    visualize_samples(torch.cat(gray_imgs), torch.cat(preds), torch.cat(color_imgs),
                      f"ImageNet class {class_id:04d} — {len(chosen_imgs)} samples")

def visualize_custom(model, folder_path=CUSTOM_DIR, n=NUM_SAMPLES):
    """Visualize samples from custom (real-world) folder."""
    # Step 1: auto-generate grayscale versions
    if not create_grayscale_versions(folder_path):
        return

    imgs = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    chosen_imgs = random.sample(imgs, min(n, len(imgs)))

    gray_imgs, color_imgs, preds = [], [], []

    for img_name in chosen_imgs:
        orig_path = os.path.join(folder_path, img_name)
        gray_path = os.path.join(GRAY_TEMP_DIR, img_name)

        gray_tensor, color_tensor = load_image(orig_path)
        gray_tensor, _ = load_image(gray_path)  # use grayscaled version as input

        gray_tensor, color_tensor = gray_tensor.to(DEVICE), color_tensor.to(DEVICE)
        with torch.no_grad():
            pred = model(gray_tensor).cpu()
        gray_imgs.append(gray_tensor.cpu())
        color_imgs.append(color_tensor.cpu())
        preds.append(pred)

    visualize_samples(torch.cat(gray_imgs), torch.cat(preds), torch.cat(color_imgs),
                      f"Custom real-world images ({len(chosen_imgs)} samples)")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ Loaded model from {MODEL_PATH}")

    print("\nChoose visualization mode:")
    print("1️⃣  Test set (by class ID)")
    print("2️⃣  Custom folder (real-world images → auto-grayscaled)")
    choice = input("Select 1 or 2: ").strip()

    if choice == "1":
        try:
            class_id = int(input("Enter class ID (0–999): "))
            visualize_class(model, class_id)
        except ValueError:
            print("❌ Invalid class ID.")
    elif choice == "2":
        folder = input(f"Enter folder path [default={CUSTOM_DIR}]: ").strip()
        if folder == "":
            folder = CUSTOM_DIR
        visualize_custom(model, folder)
    else:
        print("❌ Invalid choice. Exiting.")