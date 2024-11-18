import os
from models import UNet as Model
from piqa import PSNR, SSIM
import torch
import _utils
import numpy as np
from losses_and_regularizers import *
import torch.multiprocessing as mp
import pandas as pd
from dip_denoise import denoise
import argparse

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
psnr = PSNR()
ssim = SSIM(n_channels=1).to(dev)


def denoise_parallel(
    loss_config: CompositeLoss,
    tag: str,
    shared_data: dict,
    contaminate_with_Gaussian: bool,
):
    (
        loss_log,
        psnr_log,
        ssim_log,
        best_img,
        stop_criterion_mask_idx,
        stop_criterion_idx,
    ) = denoise(
        loss_config, shared_data, contaminate_with_Gaussian=contaminate_with_Gaussian
    )
    data = {
        "Loss": loss_log,
        "PSNR": psnr_log,
        "SSIM": ssim_log,
        "Stop_criterion_mask_idx": stop_criterion_mask_idx,
        "Stop_criterion_idx": stop_criterion_idx,
    }
    pd.DataFrame(data).to_csv(
        f"results/Brain{shared_data['idx']}/csvs/{tag}.csv", index=False
    )
    _utils.print_image(
        best_img, f"results/Brain{shared_data['idx']}/denoised_images/{tag}.png"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process an index between 1 and 4.")
    parser.add_argument(
        "--index",
        type=int,
        choices=range(1, 5),
        help="An index value (must be between 1 and 4 inclusive)",
        default=1,
    )
    parser.add_argument(
        "--noise_std",
        type=str,
        choices=["0.05", "0.10", "0.15", "0.20"],
        help="Noise strength (must be one of 0.05, 0.10, 0.15, 0.20)",
        default="0.15",
    )

    args = parser.parse_args()
    idx = args.index
    std = args.noise_std

    noisy_gt_cpu = _utils.crop_image(
        _utils.load_serialized_image(
            f".complex_test_data/noisy_brains/Contaminated_Brain{idx}_{std}.pt",
            normalize=False,
        )
    )  # image is contaminated (i.e., values not in the interval [0,1]), so we don't normalize
    _utils.print_image(
        (noisy_gt_cpu.numpy() * 255).astype(np.uint8),
        f"noisy_slice_Brain{idx}.png",
    )

    gt_cpu = _utils.crop_image(
        _utils.load_serialized_image(
            f".complex_test_data/brains/Brain{idx}.pt", normalize=False
        )
    )  # image is already normalized
    mask_cpu = _utils.crop_image(
        _utils.load_gray_image(
            f".complex_test_data/masks/Brain{idx}_mask.png", is_mask=True
        )
    )
    mask_size = (mask_cpu > 0).sum().item()

    gt_gpu = gt_cpu[None, ...].to(dev)
    mask_gpu = mask_cpu.to(dev)
    noisy_gt_gpu = noisy_gt_cpu[None, ...].to(dev)

    # Share memory for GPU tensors
    gt_gpu.share_memory_()
    noisy_gt_gpu.share_memory_()
    mask_gpu.share_memory_()

    shared_data = {
        "gt_gpu": gt_gpu,
        "noisy_gt_gpu": noisy_gt_gpu,
        "mask_gpu": mask_gpu,
        "idx": idx,
        "std": float(std),
    }
    contaminate_with_Gaussian = False  # if False, add Rician noise each DIP iteration
    test_name = f"Std{std}_Rician_TV{'_PerturbationRician' if contaminate_with_Gaussian else ''}"
    experiments = [
        (
            LossConfig(
                losses=[(Gaussian(), 1e2)],
                regularizers=[
                    (Discrete_Cosine_Transform(device=dev, p=1.0, dim=3), 1.0)
                ],
            ),
            f"{test_name}_1e2",
        ),
        (
            LossConfig(
                losses=[(Gaussian(), 1e8)],
                regularizers=[
                    (Discrete_Cosine_Transform(device=dev, p=1.0, dim=3), 1.0)
                ],
            ),
            f"{test_name}_1e8",
        ),
        (
            LossConfig(
                losses=[(Gaussian(), 1e9)],
                regularizers=[
                    (Discrete_Cosine_Transform(device=dev, p=1.0, dim=3), 1.0)
                ],
            ),
            f"{test_name}_1e9",
        ),
        (
            LossConfig(
                losses=[(Gaussian(), 1e1)],
                regularizers=[
                    (Discrete_Cosine_Transform(device=dev, p=1.0, dim=3), 1.0)
                ],
            ),
            f"{test_name}_1e1",
        ),
        (
            LossConfig(
                losses=[(Gaussian(), 1e0)],
                regularizers=[
                    (Discrete_Cosine_Transform(device=dev, p=1.0, dim=3), 1.0)
                ],
            ),
            f"{test_name}_1e0",
        ),
    ]

    mp.set_start_method("spawn")
    manager = mp.Manager()

    processes = [
        mp.Process(
            target=denoise_parallel,
            args=(
                CompositeLoss(experiment[0]),
                experiment[1],
                shared_data,
                contaminate_with_Gaussian,
            ),
            name=experiment[1],
        )
        for experiment in experiments
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
        print(f"Process {p.name} terminated with exit code {p.exitcode}")
        if p.exitcode != 0:
            continue


# import matplotlib.pyplot as plt

# for loss_list in loss_lists:
#     plt.plot(loss_list[0], label=f"{loss_list[1]}")
#     plt.yscale("symlog")
# plt.legend()
# plt.savefig(f"results/loss_results/Brain{idx}/{test_name}.pdf", bbox_inches="tight")
# plt.close()

# for psnr_list in psnr_lists:
#     plt.plot(psnr_list[0], label=f"{psnr_list[1]} (max.: {np.max(psnr_list[0]):.2f})")
# plt.legend()
# plt.savefig(f"results/psnr_results/Brain{idx}/{test_name}.pdf", bbox_inches="tight")
# plt.close()

# for ssim_list in ssim_lists:
#     plt.plot(ssim_list[0], label=f"{ssim_list[1]} (max.: {np.max(ssim_list[0]):.2f})")

# plt.legend()
# plt.savefig(f"results/ssim_results/Brain{idx}/{test_name}.pdf", bbox_inches="tight")
# plt.close()

# print()
