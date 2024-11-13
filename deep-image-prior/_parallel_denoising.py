import os
from models import UNet as Model
from piqa import PSNR, SSIM
import torch
import _utils
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import losses_and_regularizers
import torch.multiprocessing as mp
import pandas as pd

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
psnr = PSNR()
ssim = SSIM(n_channels=1).to(dev)


def denoise(loss_config: losses_and_regularizers.CompositeLoss, shared_data: dict):
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
    use_scheduler = False
    if use_scheduler:
        sch = ReduceLROnPlateau(opt, threshold=1e-4, min_lr=1e-7)

    # 🪵🪵
    loss_log, psnr_log, ssim_log = [], [], []
    best_psnr = -np.inf
    best_img = shared_data["noisy_gt_gpu"]

    for it in range(4000):
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

        if use_scheduler:
            sch.step(loss)

        loss.backward()
        opt.step()

    best_img = (
        (best_img * shared_data["mask_gpu"]).squeeze().cpu().detach().numpy() * 255
    ).astype(np.uint8)

    return loss_log, psnr_log, ssim_log, best_img


def denoise_parallel(
    loss_config: losses_and_regularizers.CompositeLoss,
    tag: str,
    shared_data: dict,
):
    loss_log, psnr_log, ssim_log, best_img = denoise(loss_config, shared_data)
    data = {"Loss": loss_log, "PSNR": psnr_log, "SSIM": ssim_log}
    pd.DataFrame(data).to_csv(f"results_{tag}.csv", index=False)
    _utils.print_image(best_img, f"best_img_{tag}.png")


if __name__ == "__main__":
    gt_path = "slice_original_cropped_192x192.png"
    mask_path = "scalar_field_cropped_192x192.png"

    gt_cpu = _utils.load_gray_image(gt_path)
    gt_cpu = (gt_cpu - gt_cpu.min()) / (gt_cpu.max() - gt_cpu.min())
    mask_cpu = _utils.load_gray_image(mask_path)

    gt_gpu = gt_cpu.to(dev)
    mask_gpu = mask_cpu.to(dev)

    mask_size = (mask_gpu > 0).sum().item()
    noisy_gt_gpu = _utils.add_rician_noise(gt_cpu, 0.15).to(dev)

    # Share memory for GPU tensors
    gt_gpu.share_memory_()
    noisy_gt_gpu.share_memory_()
    mask_gpu.share_memory_()

    shared_data = {
        "gt_gpu": gt_gpu,
        "noisy_gt_gpu": noisy_gt_gpu,
        "mask_gpu": mask_gpu,
    }

    _utils.print_image(
        (noisy_gt_gpu.squeeze(dim=0).cpu().detach().numpy() * 255).astype(np.uint8),
        "noisy_slice.png",
    )

    experiments = [
        (
            losses_and_regularizers.LossConfig(
                losses=[(losses_and_regularizers.Gaussian(), 1.0)], regularizers=[]
            ),
            "Gaussian",
        ),
        (
            losses_and_regularizers.LossConfig(
                losses=[(losses_and_regularizers.Rician(0.15), 1.0)], regularizers=[]
            ),
            "Rician",
        ),
        (
            losses_and_regularizers.LossConfig(
                losses=[(losses_and_regularizers.RicianNorm(0.15), 1.0)],
                regularizers=[],
            ),
            "Rician_norm",
        ),
    ]

    mp.set_start_method("spawn")
    manager = mp.Manager()

    processes = [
        mp.Process(
            target=denoise_parallel,
            args=(
                losses_and_regularizers.CompositeLoss(experiment[0]),
                experiment[1],
                shared_data,
            ),
            name=experiment[1],
        )
        for experiment in experiments
    ]

    for p in processes:
        p.start()
    loss_lists, psnr_lists, ssim_lists = [], [], []
    for p in processes:
        p.join()
        print(f"Process {p.name} terminated with exit code {p.exitcode}")
        if p.exitcode != 0:
            continue
        df = pd.read_csv(f"results_{p.name}.csv")
        loss_lists += [(df["Loss"].tolist(), p.name)]
        psnr_lists += [(df["PSNR"].tolist(), p.name)]
        ssim_lists += [(df["SSIM"].tolist(), p.name)]
        os.remove(f"results_{p.name}.csv")

    import matplotlib.pyplot as plt

    for psnr_list in psnr_lists:
        plt.plot(
            psnr_list[0], label=f"{psnr_list[1]} (max.: {np.max(psnr_list[0]):.2f})"
        )

    plt.legend()
    plt.savefig(f"psnr.pdf", bbox_inches="tight")
    plt.close()

    for ssim_list in ssim_lists:
        plt.plot(
            ssim_list[0], label=f"{ssim_list[1]} (max.: {np.max(ssim_list[0]):.2f})"
        )

    plt.legend()
    plt.savefig(f"ssim.pdf", bbox_inches="tight")
    plt.close()

    print()
