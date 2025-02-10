import torch
import numpy as np
import sys

sys.path.append("/mnt/data_drive/hrodrigo/mri_rician_noise/deep-image-prior")
from _utils import load_gray_image, add_rician_noise, print_image

target_shape = (256, 256)

for idx_brain, our_idx in zip([1, 2, 3, 4], [10, 11, 12, 13]):
    target_path = f"/mnt/data_drive/hrodrigo/mri_rician_noise/deep-image-prior/.brainweb_test_data/im_{our_idx}"
    img_gt = torch.load(f"brains/Brain{idx_brain}.pt", weights_only=True)
    mask = load_gray_image(f"masks/Brain{idx_brain}_mask.png", is_mask=True).squeeze()
    if img_gt.shape != target_shape:
        img_gt = img_gt[:256, :256][None, ...]  # crop
    torch.save(img_gt, f"{target_path}/gt.pt")
    if mask.shape != target_shape:
        mask = mask[:256, :256]
    print_image((mask.numpy() * 255).astype(np.uint8), f"{target_path}/mask.png")
    pass
