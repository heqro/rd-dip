from piqa import PSNR, SSIM
import torch
import _utils
import numpy as np
from losses_and_regularizers import *
from dip_denoise import denoise
from models import *
import argparse
import json
import sys

is_debugging = False
if is_debugging:
    sys.argv += [
        "--tag",
        "Test",
        "--model",
        "AttentiveUNet",
    ]

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
psnr = PSNR()
ssim = SSIM(n_channels=1).to(dev)


def load_experiment_data(
    brain_idx: int,
    noise_std: str,
    path: str = ".complex_test_data",
    print_noisy_image: bool = False,
):
    gt_cpu = _utils.crop_image(
        _utils.load_serialized_image(
            f".complex_test_data/brains/Brain{brain_idx}.pt", normalize=False
        )
    )  # image is already normalized
    mask_cpu = _utils.crop_image(
        _utils.load_gray_image(
            f".complex_test_data/masks/Brain{brain_idx}_mask.png", is_mask=True
        )
    )
    noisy_gt_cpu = _utils.crop_image(
        _utils.load_serialized_image(
            f"{path}/noisy_brains/Contaminated_Brain{brain_idx}_{noise_std}.pt",
            normalize=False,
        )  # image is contaminated (i.e., values not in the interval [0,1]), so we don't normalize
    )
    if print_noisy_image:
        _utils.print_image(
            (noisy_gt_cpu.numpy() * 255).astype(np.uint8),
            f"noisy_slice_Brain{brain_idx}_{noise_std}.png",
        )
    return noisy_gt_cpu, gt_cpu, mask_cpu


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
parser.add_argument(
    "--contaminate_with_Rician_noise",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether or not we contaminate with Rician noise for each DIP iteration. Defaults to False.",
)
parser.add_argument(
    "--fidelities",
    type=str,
    nargs="+",
    help=(
        "List of loss functions with weights and parameters in the format "
        "'LossName:weight:param1:param2...'. Example: 'Gaussian:1.0', 'Rician:0.5:0.1'"
    ),
    default=[
        "Rician:0.5:0.15",
        "Gaussian:0.1",
    ],  # TODO blows up for 0.10 in Rician term?
    required=not is_debugging,
)
parser.add_argument(
    "--regularizers",
    type=str,
    nargs="*",
    help=(
        "List of regularization terms with weights and parameters in the format "
        "'RegName:weight:param1:param2...'. Example: 'Total_Variation:0.1:1.0', "
        "'Discrete_Cosine_Transform:0.2:3:1.0:1e-6'"
    ),
    default=[],
)
parser.add_argument(
    "--tag",
    type=str,
    required=not is_debugging,
    help=(
        "The tag for the experiment. It should reflect the characteristics of it for finding it easily along other results. An idea is Std$std_$Fidelities_$Regularizers"
    ),
)
parser.add_argument("--model", type=str, required=not is_debugging)


# ⌨️
args = parser.parse_args()
model = globals().get(args.model)

if model is None:
    print(f"Model {args.model} not found")
    quit(-1)
model = model()  # initialize model with default parameters
idx = args.index
std = args.noise_std
contaminate_with_Rician = args.contaminate_with_Rician_noise
composite_loss = load_experiment_config(args.fidelities, args.regularizers)
# 🖼️
noisy_gt_cpu, gt_cpu, mask_cpu = load_experiment_data(brain_idx=idx, noise_std=std)
mask_size = (mask_cpu > 0).sum().item()
gt_gpu = gt_cpu[None, ...].to(dev)
mask_gpu = mask_cpu.to(dev)
noisy_gt_gpu = noisy_gt_cpu[None, ...].to(dev)

shared_data = {
    "gt_gpu": gt_gpu,
    "noisy_gt_gpu": noisy_gt_gpu,
    "mask_gpu": mask_gpu,
    "idx": idx,
    "std": float(std),
    "model": model,
}
report, best_img = denoise(
    composite_loss, shared_data, contaminate_with_Gaussian=not contaminate_with_Rician
)

_utils.print_image(
    best_img, f"results/Brain{shared_data['idx']}/denoised_images/{args.tag}.png"
)

with open(f"results/Brain{shared_data['idx']}/jsons/{args.tag}.json", "w") as json_file:
    json.dump(report, json_file, indent=4)
