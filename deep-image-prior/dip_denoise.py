from losses_and_regularizers import CompositeLoss
import torch
import numpy as np
from models import UNet as Model
import _utils
from piqa import PSNR, SSIM
from torch.nn.functional import mse_loss as MSE
from torch import Tensor


def denoise(
    loss_config: CompositeLoss, shared_data: dict, contaminate_with_Gaussian=True
):
    def stopping_criterion(prediction: Tensor, noisy_img: Tensor, std: float):
        omega = 1
        for dim in prediction.shape:
            omega = omega * dim
        return MSE(prediction, noisy_img) < std**2

    def stopping_criterion_mask(
        prediction: Tensor, noisy_img: Tensor, mask: Tensor, std: float
    ):
        omega = (mask > 0).sum().item()
        prediction = prediction * mask
        noisy_img = noisy_img * mask
        return ((prediction - noisy_img) * mask).square().sum() / omega < std**2

    dev = shared_data["gt_gpu"].device
    psnr = PSNR()
    ssim = SSIM(n_channels=1).to(dev)
    model = Model(n_channels_output=1).to(dev)
    seed_cpu = torch.from_numpy(
        np.random.uniform(
            0,
            0.1,
            size=(3, shared_data["gt_gpu"].shape[-2], shared_data["gt_gpu"].shape[-1]),
        ).astype("float32")
    )[None, :]
    seed_gpu = seed_cpu.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    # 🪵🪵
    loss_log, psnr_log, ssim_log = [], [], []
    best_psnr = -np.inf
    best_img = shared_data["noisy_gt_gpu"]
    reached_stopping_criterion = reached_stopping_criterion_mask = False
    stopping_criterion_idx = stopping_criterion_mask_idx = None

    for it in range(10000):
        opt.zero_grad()
        noisy_seed_dev = (
            _utils.add_gaussian_noise(seed_gpu, avg=0, std=0.05)
            if contaminate_with_Gaussian
            else _utils.add_rician_noise(seed_gpu, std=0.15)
        )
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
        if not reached_stopping_criterion:
            if stopping_criterion(
                prediction[0], shared_data["noisy_gt_gpu"], shared_data["std"]
            ):
                reached_stopping_criterion = True
                stopping_criterion_idx = it
        if not reached_stopping_criterion_mask:
            if stopping_criterion_mask(
                prediction[0],
                shared_data["noisy_gt_gpu"],
                shared_data["mask_gpu"],
                shared_data["std"],
            ):
                reached_stopping_criterion_mask = True
                stopping_criterion_mask_idx = it
        loss.backward()
        opt.step()

    best_img = (
        (best_img * shared_data["mask_gpu"]).squeeze().cpu().detach().numpy() * 255
    ).astype(np.uint8)

    return (
        loss_log,
        psnr_log,
        ssim_log,
        best_img,
        stopping_criterion_mask_idx,
        stopping_criterion_idx,
    )
