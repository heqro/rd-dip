from models import UNet as Model
from piqa import PSNR, SSIM
import torch
import _utils
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import losses_and_regularizers
import torch.multiprocessing as mp

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

    for it in range(5000):
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
    output_dict: dict,
    idx: int,
    shared_data: dict,
):
    loss_log, psnr_log, ssim_log, best_img = denoise(loss_config, shared_data)
    output_dict[idx] = (loss_log, psnr_log, ssim_log, best_img)


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

    gaussian_loss = losses_and_regularizers.LossConfig(
        losses=[(losses_and_regularizers.Gaussian(), 1.0)], regularizers=[]
    )
    rician_loss = losses_and_regularizers.LossConfig(
        losses=[(losses_and_regularizers.Rician(0.15), 1.0)], regularizers=[]
    )
    rician_norm_loss = losses_and_regularizers.LossConfig(
        losses=[(losses_and_regularizers.RicianNorm(0.15), 1.0)], regularizers=[]
    )

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    output_dict = manager.dict()

    processes = [
        mp.Process(
            target=denoise_parallel,
            args=(
                losses_and_regularizers.CompositeLoss(gaussian_loss),
                output_dict,
                0,
                shared_data,
            ),
        ),
        mp.Process(
            target=denoise_parallel,
            args=(
                losses_and_regularizers.CompositeLoss(rician_loss),
                output_dict,
                1,
                shared_data,
            ),
        ),
        mp.Process(
            target=denoise_parallel,
            args=(
                losses_and_regularizers.CompositeLoss(rician_norm_loss),
                output_dict,
                2,
                shared_data,
            ),
        ),
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    loss_gaussian, psnr_gaussian, ssim_gaussian, best_img_gaussian = output_dict[0]
    loss_rician, psnr_rician, ssim_rician, best_img_rician = output_dict[1]
    loss_rician_norm, psnr_rician_norm, ssim_rician_norm, best_img_rician_norm = (
        output_dict[2]
    )

    import matplotlib.pyplot as plt

    plt.plot(psnr_gaussian, label=f"Gaussian (max.: {np.max(psnr_gaussian):.2f})")
    plt.plot(psnr_rician, label=f"Rician (max.: {np.max(psnr_rician):.2f})")
    plt.plot(
        psnr_rician_norm, label=f"Rician norm (max.: {np.max(psnr_rician_norm):.2f})"
    )
    plt.legend()
    plt.savefig(f"psnr.pdf", bbox_inches="tight")
    plt.close()

    plt.plot(ssim_gaussian, label=f"Gaussian (max.: {np.max(ssim_gaussian):.2f})")
    plt.plot(ssim_rician, label=f"Rician (max.: {np.max(ssim_rician):.2f})")
    plt.plot(
        ssim_rician_norm, label=f"Rician norm (max.: {np.max(ssim_rician_norm):.2f})"
    )
    plt.legend()
    plt.savefig(f"ssim.pdf", bbox_inches="tight")
    plt.close()

    _utils.print_image(best_img_gaussian, "best_img_gaussian.png")
    _utils.print_image(best_img_rician, "best_img_rician.png")
    _utils.print_image(best_img_rician_norm, "best_img_rician_norm.png")

    print()
