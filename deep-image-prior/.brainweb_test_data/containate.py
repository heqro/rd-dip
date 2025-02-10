import numpy as np
import torch
import sys
import os

sys.path.append("/mnt/data_drive/hrodrigo/mri_rician_noise/deep-image-prior")
from _utils import load_gray_image, add_rician_noise, print_image


# noise = "0.20"
for noise in ["0.05", "0.10", "0.15", "0.20"]:
    for i in range(10, 14):
        img_gt = torch.load(f"im_{i}/gt.pt", weights_only=False)
        print(f"{i} dim: {img_gt.shape}")
        noisy = add_rician_noise(img_gt, float(noise))
        print_image((noisy.numpy()[0] * 255).astype(np.uint8), f"im_{i}/Std{noise}.png")
        torch.save(noisy, f"im_{i}/Std{noise}.pt")
