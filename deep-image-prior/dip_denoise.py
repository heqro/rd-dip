from losses_and_regularizers import CompositeLoss
import torch
import numpy as np
from models import UNet as Model
import _utils
from piqa import PSNR, SSIM
from torch.nn.functional import mse_loss as MSE
from torch import Tensor
from typing import Tuple, TypedDict


class StoppingCriteria(TypedDict):
    mask_idx: int | None
    entire_image_idx: int | None


class LossAddend(TypedDict):
    # Fidelities and Regularizers go in here
    coefficient: float
    values: list[float]


class LossLog(TypedDict):
    overall_loss: list[float]
    addends: dict[str, LossAddend]


class ExperimentReport(TypedDict):
    # Fields for results
    psnr_entire_image_log: list[float]
    psnr_mask_log: list[float]
    ssim_mask_log: list[float]
    stopping_criteria_indices: StoppingCriteria
    loss_log: LossLog
    # Data of the experiment
    contaminate_with_Gaussian: bool
    std: float


def initialize_experiment_report(
    contaminate_with_Gaussian: bool, std: float
) -> ExperimentReport:
    return {
        "contaminate_with_Gaussian": contaminate_with_Gaussian,
        "loss_log": {"overall_loss": [], "addends": {}},
        "psnr_entire_image_log": [],
        "psnr_mask_log": [],
        "ssim_mask_log": [],
        "std": std,
        "stopping_criteria_indices": {"entire_image_idx": None, "mask_idx": None},
    }


def denoise(
    loss_config: CompositeLoss, shared_data: dict, contaminate_with_Gaussian=True
) -> Tuple[ExperimentReport, np.ndarray]:
    def stopping_criterion(prediction: Tensor, noisy_img: Tensor, std: float):
        return MSE(prediction, noisy_img) < std**2

    def stopping_criterion_mask(
        prediction: Tensor, noisy_img: Tensor, mask: Tensor, std: float
    ):
        omega = (mask > 0).sum().item()
        prediction = prediction * mask
        noisy_img = noisy_img * mask
        return ((prediction - noisy_img) * mask).square().sum() / omega < std**2

    experiment_report = initialize_experiment_report(
        contaminate_with_Gaussian=contaminate_with_Gaussian, std=shared_data["std"]
    )
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
    best_psnr = -np.inf
    best_img = shared_data["noisy_gt_gpu"]
    reached_stopping_criterion = reached_stopping_criterion_mask = False

    for it in range(10000):
        opt.zero_grad()
        noisy_seed_dev = (
            _utils.add_gaussian_noise(seed_gpu, avg=0, std=0.05)
            if contaminate_with_Gaussian
            else _utils.add_rician_noise(seed_gpu, std=0.15)
        )
        prediction = model.forward(noisy_seed_dev)
        if prediction.isnan().any():
            print("ERR: model.forward() returned NANs")
        loss = loss_config.evaluate_losses(prediction[0], shared_data["noisy_gt_gpu"])

        # 🪵
        experiment_report["psnr_entire_image_log"] += [
            psnr(prediction[0], shared_data["gt_gpu"]).item()
        ]
        experiment_report["ssim_mask_log"] += [
            ssim(
                (prediction[0] * shared_data["mask_gpu"]).permute(1, 0, 2),
                (shared_data["gt_gpu"] * shared_data["mask_gpu"]).permute(1, 0, 2),
            ).item()
        ]
        it_psnr_mask = _utils.psnr_with_mask(
            prediction[0], shared_data["gt_gpu"], shared_data["mask_gpu"]
        ).item()
        experiment_report["psnr_mask_log"] += [it_psnr_mask]

        if best_psnr < it_psnr_mask:
            best_psnr = it_psnr_mask
            best_img = prediction
        if not reached_stopping_criterion:
            if stopping_criterion(
                prediction[0], shared_data["noisy_gt_gpu"], shared_data["std"]
            ):
                reached_stopping_criterion = True
                experiment_report["stopping_criteria_indices"]["entire_image_idx"] = it
        if not reached_stopping_criterion_mask:
            if stopping_criterion_mask(
                prediction[0],
                shared_data["noisy_gt_gpu"],
                shared_data["mask_gpu"],
                shared_data["std"],
            ):
                reached_stopping_criterion_mask = True
                experiment_report["stopping_criteria_indices"]["mask_idx"] = it
        loss.backward()
        opt.step()

    best_img = (
        (best_img * shared_data["mask_gpu"]).squeeze().cpu().detach().numpy() * 255
    ).astype(np.uint8)

    experiment_report["loss_log"]["overall_loss"] = loss_config.total_loss_log
    for fid_fn, lambda_loss in loss_config.config["fidelities"]:
        experiment_report["loss_log"]["addends"][fid_fn.__class__.__name__] = {
            "coefficient": lambda_loss,
            "values": loss_config.fid_loss_log[fid_fn.__class__.__name__],
        }
    for reg_fn, lambda_loss in loss_config.config["regularizers"]:
        experiment_report["loss_log"]["addends"][reg_fn.__class__.__name__] = {
            "coefficient": lambda_loss,
            "values": loss_config.reg_loss_log[reg_fn.__class__.__name__],
        }
    return experiment_report, best_img
