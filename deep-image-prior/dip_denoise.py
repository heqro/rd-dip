from losses_and_regularizers import CompositeLoss
import torch
import numpy as np
import _utils
from piqa import PSNR, SSIM
from torch.nn.functional import mse_loss as MSE
from torch import Tensor, nn
from typing import Callable, Literal, Tuple, TypedDict


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


class DIP_Report(TypedDict):
    it_noise_type: Literal["", "Rician", "Gaussian"]
    it_noise_std: float
    net_architecture: dict[str, str]
    simultaneous_perturbations: int


class OptimizerProfile(TypedDict):
    name: str
    lr: list[float]


class ExperimentReport(TypedDict):
    # Fields for results
    exit_code: int
    name: str
    psnr_entire_image_log: list[float]
    psnr_mask_log: list[float]
    ssim_mask_log: list[float]
    ssim_entire_image_log: list[float]
    loss_log: LossLog
    stopping_criteria_indices: StoppingCriteria
    optimizer: OptimizerProfile
    # Fields for the image
    image_noise_std: float
    # DIP data
    dip_config: DIP_Report


class ProblemImages(torch.nn.Module):

    def __init__(
        self,
        ground_truth: Tensor,
        noisy_image: Tensor,
        mask: Tensor,
        rician_noise_std: float,
    ):
        super().__init__()
        self.ground_truth = nn.Parameter(ground_truth, requires_grad=False)
        self.noisy_image = nn.Parameter(noisy_image, requires_grad=False)
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.rician_noise_std = rician_noise_std


class DIP(TypedDict):
    noise_fn: Callable[..., Tensor]
    std: float
    model: nn.Module
    seed: Tensor
    simultaneous_perturbations: int


class Problem(TypedDict):
    image_name: str
    tag: str
    images: ProblemImages
    optimizer: torch.optim.Optimizer
    loss_config: CompositeLoss
    dip_config: DIP
    max_its: int
    psnr: PSNR
    ssim: SSIM


def initialize_experiment_report(p: Problem) -> ExperimentReport:
    aux = ""
    if p["dip_config"]["noise_fn"] == _utils.add_rician_noise:
        aux = "Rician"
    if p["dip_config"]["noise_fn"] == _utils.add_gaussian_noise:
        aux = "Gaussian"
    return {
        "exit_code": 0,
        "name": p["image_name"],
        "dip_config": {
            "it_noise_type": aux,
            "it_noise_std": p["dip_config"]["std"],
            "simultaneous_perturbations": p["dip_config"]["simultaneous_perturbations"],
            "net_architecture": {
                "name": p["dip_config"]["model"]._get_name(),
                "channels_list": p["dip_config"]["model"].channels_list,
                "skips_sizes": p["dip_config"]["model"].skip_sizes,
            },
        },
        "image_noise_std": p["images"].rician_noise_std,
        "loss_log": {"overall_loss": [], "addends": {}},
        "psnr_entire_image_log": [],
        "psnr_mask_log": [],
        "ssim_mask_log": [],
        "ssim_entire_image_log": [],
        "stopping_criteria_indices": {"entire_image_idx": None, "mask_idx": None},
        "optimizer": {
            "lr": [p["optimizer"].param_groups[-1]["lr"]],
            "name": type(p["optimizer"]).__name__,
        },
    }


def stopping_criterion(prediction: Tensor, noisy_img: Tensor, std: float):
    return MSE(prediction, noisy_img) < std**2


def stopping_criterion_mask(
    prediction: Tensor, noisy_img: Tensor, mask: Tensor, std: float
):
    omega = (mask > 0).sum().item()
    prediction = prediction * mask
    noisy_img = noisy_img * mask
    return ((prediction - noisy_img) * mask).square().sum() / omega < std**2


def update_report_quality_metrics(
    report: ExperimentReport,
    it: int,
    prediction: Tensor,
    p: Problem,
    best_psnr: float,
    bbox: tuple[tuple[int, int], tuple[int, int]],
    best_img: Tensor,
):
    # Update PSNR
    report["psnr_entire_image_log"] += [
        p["psnr"](prediction, p["images"].ground_truth).item()
    ]
    it_psnr_mask = _utils.psnr_with_mask(
        prediction, p["images"].ground_truth, p["images"].mask
    ).item()
    report["psnr_mask_log"] += [it_psnr_mask]

    if best_psnr < it_psnr_mask:
        best_psnr = it_psnr_mask
        best_img = prediction

    # Update SSIM
    report["ssim_entire_image_log"] += [
        p["ssim"](
            prediction,
            p["images"].ground_truth,
        ).item()
    ]
    report["ssim_mask_log"] += [
        p["ssim"](
            (prediction * p["images"].mask)[
                ...,
                bbox[0][0] : bbox[1][0],
                bbox[0][1] : bbox[1][1],
            ],
            (p["images"].ground_truth * p["images"].mask)[
                ...,
                bbox[0][0] : bbox[1][0],
                bbox[0][1] : bbox[1][1],
            ],
        ).item()
    ]

    # Verify stopping criterions
    if report["stopping_criteria_indices"][
        "entire_image_idx"
    ] is None and stopping_criterion(
        prediction, p["images"].noisy_image, p["images"].rician_noise_std
    ):
        report["stopping_criteria_indices"]["entire_image_idx"] = it

    if report["stopping_criteria_indices"][
        "mask_idx"
    ] is None and stopping_criterion_mask(
        prediction,
        p["images"].noisy_image,
        p["images"].mask,
        p["images"].rician_noise_std,
    ):
        report["stopping_criteria_indices"]["mask_idx"] = it

    return best_psnr, best_img


def update_report_losses(
    experiment_report: ExperimentReport, loss_config: CompositeLoss
):
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


def solve(
    p: Problem, experiment_report: ExperimentReport
) -> Tuple[ExperimentReport, Tensor]:

    model = p["dip_config"]["model"]
    seed = p["dip_config"]["seed"]
    std = p["dip_config"]["std"]
    # 🪵🪵
    best_psnr = -np.inf
    best_img = p["images"].noisy_image
    bounding_box = _utils.get_bounding_box(p["images"].mask)

    for it in range(p["max_its"]):
        p["optimizer"].zero_grad()
        noisy_seed = p["dip_config"]["noise_fn"](seed, std=std)
        prediction = (
            model.forward(noisy_seed).clip(0, 1).mean(dim=0)[None, ...]
        )  # mean is no-op when batch_size == 1

        if prediction.isnan().any():
            print("ERR: model.forward() returned NANs. Terminating prematurely.")
            experiment_report["exit_code"] = -1
            break

        loss = p["loss_config"].evaluate_losses(prediction, p["images"].noisy_image)

        if loss.isnan():
            print("ERR: loss is NAN. Terminating prematurely.")
            experiment_report["exit_code"] = -1
            break

        best_psnr, best_img = update_report_quality_metrics(
            experiment_report,
            it,
            prediction,
            p,
            best_psnr,
            bounding_box,
            best_img,
        )
        loss.backward()
        p["optimizer"].step()
    update_report_losses(experiment_report, p["loss_config"])
    return experiment_report, best_img
