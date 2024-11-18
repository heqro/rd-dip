import numpy as np
import torch
from piqa import PSNR, SSIM
import _utils
from PIL import Image


noisy_gt_exists = True
gt_path = "slice_original_cropped_192x192.png"
mask_path = "scalar_field_cropped_192x192.png"

gt_cpu = _utils.load_gray_image(gt_path)
gt_cpu = (gt_cpu - gt_cpu.min()) / (gt_cpu.max() - gt_cpu.min())
mask_cpu = _utils.load_gray_image(mask_path, is_mask=True)

mask_size = (mask_cpu > 0).sum().item()
if not noisy_gt_exists:
    noisy_gt_cpu = _utils.add_rician_noise(gt_cpu, 0.15)
else:
    noisy_gt_cpu = torch.load("noisy_gt_cpu.pt")
noisy_gt_cpu = noisy_gt_cpu.numpy()

import bm3d

std = 0.15
denoised_image = bm3d.bm3d(
    2 * np.sqrt(noisy_gt_cpu[0] + std**2) - std**2, sigma_psd=std
)
denoised_image = (denoised_image + std**2) ** 2 / 4 - std**2
psnr_result = _utils.psnr_with_mask(denoised_image, gt_cpu[0].numpy(), mask_cpu.numpy())
new_p = Image.fromarray(((denoised_image * mask_cpu.numpy()[0]) * 255).astype(np.uint8))
new_p = new_p.convert("L")
_utils.print_image(new_p, f"results/denoised_images/best_img_bm3d_nomask.png")
print(psnr_result)
