from losses_and_regularizers import CompositeLoss
import torch
import numpy as np
from models import UNet as Model
import _utils
from piqa import PSNR, SSIM


def denoise(loss_config: CompositeLoss, shared_data: dict):
    dev = shared_data["gt_gpu"].device
    psnr = PSNR()
    ssim = SSIM(n_channels=1).to(dev)
    model = Model(n_channels_output=1).to(dev)
    seed_cpu = torch.from_numpy(
        np.random.uniform(
            0,
            0.1,
            size=(3, shared_data["gt_gpu"].shape[1], shared_data["gt_gpu"].shape[2]),
        ).astype("float32")
    )[None, :]
    seed_gpu = seed_cpu.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    # 🪵🪵
    loss_log, psnr_log, ssim_log = [], [], []
    best_psnr = -np.inf
    best_img = shared_data["noisy_gt_gpu"]

    for it in range(10000):
        opt.zero_grad()
        noisy_seed_dev = _utils.add_gaussian_noise(seed_gpu, avg=0, std=0.05)
        prediction = model.forward(noisy_seed_dev)
        loss = loss_config.forward(prediction[0], shared_data["noisy_gt_gpu"])

        it_psnr = psnr(prediction[0], shared_data["gt_gpu"]).item()
        it_psnr_mask = _utils.psnr_with_mask(
            prediction[0], shared_data["gt_gpu"], shared_data["mask_gpu"]
        ).item()
        it_ssim = ssim(
            (prediction[0] * shared_data["mask_gpu"]).permute(1, 0, 2),
            (shared_data["gt_gpu"] * shared_data["mask_gpu"]).permute(1, 0, 2),
        ).item()

        loss_log += [loss.item()]
        psnr_log += [it_psnr_mask]
        ssim_log += [it_ssim]

        if best_psnr < it_psnr_mask:
            best_psnr = it_psnr_mask
            best_img = prediction

        loss.backward()
        opt.step()

    best_img = (
        (best_img * shared_data["mask_gpu"]).squeeze().cpu().detach().numpy() * 255
    ).astype(np.uint8)

    return loss_log, psnr_log, ssim_log, best_img
