import torch
from torchmetrics.functional import structural_similarity_index_measure as ssim
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir,'..'))
if project_root not in sys.path :
    sys.path.append(project_root)
from scripts.train64 import UNet
from scripts.images64_split import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = UNet().to(device)
MODELDIR="results/checkpoints/best_unet_colorizer.pt"
model.load_state_dict(torch.load(MODELDIR))
model.eval()

# Load validation data
_, val_loader, _ = get_dataloaders(batch_size=512)

# Compute L1 and SSIM
l1_loss = torch.nn.L1Loss()
val_l1, val_ssim = 0.0, 0.0

with torch.no_grad():
    for gray, color in val_loader:
        gray, color = gray.to(device), color.to(device)
        pred = model(gray)
        val_l1 += l1_loss(pred, color).item()
        val_ssim += ssim(pred, color).item()

val_l1 /= len(val_loader)
val_ssim /= len(val_loader)

print(f"Model: ",MODELDIR)
print(f"Validation L1: {val_l1:.4f}")
print(f"Validation SSIM: {val_ssim:.4f}")