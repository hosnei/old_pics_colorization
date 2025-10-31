#!/usr/bin/env python3
"""
plot_finetune_curves_1class.py
==============================
Plots training and validation L1 + validation SSIM for the 1-class fine-tuning run.
"""

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Data from your logs (1 class, 50 epochs)
# ---------------------------------------------------------------------
epochs = list(range(1, 51))

train_l1 = [
    0.3837, 0.2506, 0.2017, 0.1730, 0.1551, 0.1486, 0.1417, 0.1407, 0.1358, 0.1336,
    0.1410, 0.1328, 0.1298, 0.1307, 0.1294, 0.1312, 0.1318, 0.1284, 0.1287, 0.1287,
    0.1280, 0.1295, 0.1264, 0.1241, 0.1280, 0.1236, 0.1238, 0.1250, 0.1321, 0.1273,
    0.1261, 0.1252, 0.1236, 0.1218, 0.1234, 0.1289, 0.1296, 0.1218, 0.1235, 0.1259,
    0.1228, 0.1235, 0.1229, 0.1249, 0.1197, 0.1198, 0.1199, 0.1205, 0.1242, 0.1241
]

val_l1 = [
    0.4647, 0.3890, 0.2623, 0.1875, 0.1550, 0.1386, 0.1326, 0.1300, 0.1273, 0.1254,
    0.1258, 0.1232, 0.1216, 0.1296, 0.1206, 0.1214, 0.1214, 0.1191, 0.1188, 0.1187,
    0.1196, 0.1239, 0.1171, 0.1170, 0.1192, 0.1182, 0.1164, 0.1177, 0.1181, 0.1156,
    0.1161, 0.1168, 0.1163, 0.1180, 0.1184, 0.1305, 0.1170, 0.1150, 0.1192, 0.1164,
    0.1147, 0.1179, 0.1176, 0.1214, 0.1175, 0.1147, 0.1202, 0.1831, 0.1840, 0.1270
]

val_ssim = [
    0.0288, 0.1603, 0.4226, 0.5598, 0.6091, 0.6542, 0.6690, 0.6762, 0.6897, 0.6940,
    0.6954, 0.7021, 0.7089, 0.6973, 0.7125, 0.7096, 0.7132, 0.7168, 0.7187, 0.7201,
    0.7189, 0.7122, 0.7234, 0.7255, 0.7182, 0.7216, 0.7259, 0.7256, 0.7284, 0.7284,
    0.7289, 0.7297, 0.7263, 0.7281, 0.7227, 0.6968, 0.7255, 0.7306, 0.7189, 0.7289,
    0.7310, 0.7235, 0.7248, 0.7175, 0.7214, 0.7342, 0.7242, 0.6192, 0.6483, 0.7044
]

# ---------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 5))

# --- L1 Loss ---
ax1.set_xlabel("Epoch")
ax1.set_ylabel("L1 Loss (↓)", color="tab:blue")
ax1.plot(epochs, train_l1, marker='^', color='tab:cyan', label="Train L1")
ax1.plot(epochs, val_l1, marker='o', color='tab:blue', label="Val L1")
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax1.invert_yaxis()  # smaller = better
ax1.grid(True, linestyle='--', alpha=0.3)

# --- SSIM ---
ax2 = ax1.twinx()
ax2.set_ylabel("Validation SSIM (↑)", color="tab:orange")
ax2.plot(epochs, val_ssim, marker='s', color="tab:orange", label="Val SSIM")
ax2.tick_params(axis='y', labelcolor="tab:orange")

# --- Legends ---
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")

plt.title("Training vs Validation Curves — U-Net Colorizer (1 Class)")
fig.tight_layout()
plt.show()