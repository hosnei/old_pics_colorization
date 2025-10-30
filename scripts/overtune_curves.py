#!/usr/bin/env python3
"""
plot_finetune_curves.py
=======================
Plots validation L1 and SSIM evolution over fine-tuning epochs.
"""

import matplotlib.pyplot as plt

# --- Paste your collected results here ---
epochs     = [0, 1, 2, 3, 4, 5]   # 0 = baseline before fine-tuning
val_l1     = [0.0933, 0.0921, 0.0921, 0.0920, 0.0920, 0.0921]
val_ssim   = [0.7854, 0.7899, 0.7899, 0.7900, 0.7900, 0.7897]

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(8, 5))

color = 'tab:blue'
ax1.set_xlabel('Fine-tuning Epoch')
ax1.set_ylabel('Validation L1 Loss (↓)', color=color)
ax1.plot(epochs, val_l1, marker='o', color=color, label='L1 Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.invert_yaxis()  # smaller L1 = better

ax2 = ax1.twinx()  # second axis for SSIM
color = 'tab:orange'
ax2.set_ylabel('Validation SSIM (↑)', color=color)
ax2.plot(epochs, val_ssim, marker='s', color=color, label='SSIM')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Fine-tuning Over-tuning Curves — UNet Colorizer')
fig.tight_layout()
plt.show()