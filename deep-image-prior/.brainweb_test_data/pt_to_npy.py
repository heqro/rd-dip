import numpy as np
import torch
import sys
import os

sys.path.append("/mnt/data_drive/hrodrigo/mri_rician_noise/deep-image-prior")
from _utils import load_gray_image, add_rician_noise, print_image


# noise = "0.20"
for i in range(1, 11):
    img_gt = torch.load(f"im_{i}/gt.pt", weights_only=False)
    np.save(file=f"im_{i}/gt", arr=img_gt.numpy())
    for noise in ["0.05", "0.10", "0.15", "0.20"]:
        img_noisy = torch.load(f"im_{i}/Std{noise}.pt", weights_only=False)
        np.save(file=f"im_{i}/Std{noise}", arr=img_noisy.numpy())
