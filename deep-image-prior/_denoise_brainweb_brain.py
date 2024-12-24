from typing import Literal
from piqa import PSNR, SSIM
import torch
import _utils
import numpy as np
from losses_and_regularizers import *
from dip_denoise import Problem, ProblemImages, solve, initialize_experiment_report
from skimage.metrics import structural_similarity as ssim_sk
from models import *
import argparse
import json
import sys


is_debugging = False
if is_debugging:
    sys.argv += [
        "--subject",
        "04",
        "--noise_std",
        "0.15",
        "--fidelities",
        "Rician_Norm:1.0:0.15",
        "--regularizers",
        "Total_Variation:1.0:1.0:0.75",
        "--tag",
        "Test",
        "--max_its",
        "30",
        "--dip_noise_type",
        "Rician",
        "--dip_noise_std",
        "0.05",
        "--model",
        "UNet",
        "--lr",
        "1e-3",
    ]


dev = (
    torch.device("cuda")
    if torch.cuda.is_available()  # and not is_debugging
    else torch.device("cpu")
)
psnr = PSNR()
ssim = SSIM(n_channels=1).to(dev)


def save_best_img(best_img: Tensor, it: str, subject_idx: str):
    _utils.print_image(
        (best_img.squeeze().cpu().detach().numpy() * 255).astype(np.uint8),
        f"results/subject_{subject_idx}/def_denoised/{args.tag}{it}.png",
    )


def save_best_ssim(best_img: Tensor, ground_truth: Tensor, it: str, subject_idx: str):
    _, img_ssim = ssim_sk(
        ground_truth.squeeze().cpu().numpy(),
        best_img.squeeze().cpu().detach().numpy(),
        data_range=1.0,
        channel_axis=0,
        full=True,
    )  # img_ssim is HxW
    _utils.print_image(
        (img_ssim.clip(0, 1) * 255).astype(np.uint8),
        f"results/subject_{subject_idx}/def_ssim/{args.tag}{it}.png",
    )


def get_dip_noise_fn(noise_type: Literal["", "Gaussian", "Rician"]):
    if noise_type == "Rician":
        return _utils.add_rician_noise
    if noise_type == "Gaussian":
        return _utils.add_gaussian_noise

    def id(x: Tensor):
        return x

    return id


def load_experiment_data(
    subject_idx: str,
    noise_std: str,
    print_noisy_image: bool = False,
):
    gt_cpu = _utils.crop_image(
        _utils.load_serialized_image(
            f".brainweb_test_data/subject_{subject_idx}/subject_{subject_idx}_gt.pt",
            normalize=False,
        )
    )  # image is already normalized
    mask_cpu = _utils.crop_image(
        _utils.load_gray_image(
            f".brainweb_test_data/subject_{subject_idx}/subject_{subject_idx}_mask.png",
            is_mask=True,
        )
    )
    noisy_gt_cpu = _utils.crop_image(
        _utils.load_serialized_image(
            f".brainweb_test_data/subject_{subject_idx}/subject_{subject_idx}_Std{noise_std}.pt",
            normalize=False,
        )  # image is contaminated (i.e., values not in the interval [0,1]), so we don't normalize
    )
    if print_noisy_image:
        _utils.print_image(
            (noisy_gt_cpu.numpy() * 255).astype(np.uint8),
            f"noisy_slice_subject_{subject_idx}_Std{noise_std}.png",
        )
    return noisy_gt_cpu, gt_cpu, mask_cpu


parser = argparse.ArgumentParser()
parser.add_argument(
    "--subject",
    type=str,
    help="The brain subject",
    default="04",
)
parser.add_argument(
    "--noise_std",
    type=str,
    choices=["0.05", "0.10", "0.15", "0.20"],
    help="Noise strength (must be one of 0.05, 0.10, 0.15, 0.20)",
    default="0.15",
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
        # "Rician:0.5:0.15",
        "Gaussian:1.0",
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
parser.add_argument(
    "--dip_noise_type", choices=["Gaussian", "Rician", ""], default="Gaussian"
)
parser.add_argument("--max_its", type=int, default=30000)
parser.add_argument("--dip_noise_std", type=float, default=0.15)
parser.add_argument("--model", type=str, required=not is_debugging)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--N", type=int, default=1)


# ⌨️
args = parser.parse_args()
model = globals().get(args.model)

if model is None:
    print(f"Model {args.model} not found")
    quit(-1)

model = model().to(dev)  # initialize model with default parameters
std = args.noise_std
composite_loss = load_experiment_config(args.fidelities, args.regularizers)
noisy_gt_cpu, gt_cpu, mask_cpu = load_experiment_data(
    subject_idx=args.subject, noise_std=std
)

images = ProblemImages(
    ground_truth=gt_cpu[None, ...],
    noisy_image=noisy_gt_cpu[None, ...],
    mask=mask_cpu[None, ...],
    rician_noise_std=float(std),
).to(dev)

seed = 0.1 * torch.rand(
    1,
    3,
    images.mask.shape[-2],
    images.mask.shape[-1],
).to(
    images.noisy_image.device
).expand(args.N, -1, -1, -1)

p: Problem = {
    "images": images,
    "psnr": psnr,
    "ssim": ssim,
    "max_its": args.max_its,
    "tag": args.tag,
    "optimizer": torch.optim.Adam(model.parameters(), lr=args.lr),
    "loss_config": composite_loss,
    "dip_config": {
        "noise_fn": get_dip_noise_fn(args.dip_noise_type),
        "std": args.dip_noise_std,
        "model": model,
        "seed": seed,
    },
    "image_name": f"subject_{args.subject}",
}

report = initialize_experiment_report(p)

# for it in range(1, 3000):  # take 3000 'best images'
report, best_img = solve(p, report)
save_best_img(best_img, "", subject_idx=args.subject)
save_best_ssim(best_img, images.ground_truth, "", subject_idx=args.subject)
# if report["exit_code"] != 0:
#     break


with open(
    f"results/subject_{args.subject}/def_jsons/{args.tag}.json", "w"
) as json_file:
    json.dump(report, json_file, indent=4)
