from typing import Literal
from piqa import PSNR, SSIM
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_sk
import argparse
import json
import sys

sys.path.append("../libs/image-utils/src")
from my_io import print_image, load_serialized_image, load_gray_image
from noise import add_gaussian_noise, add_rician_noise
from spatial_transforms import crop_image

sys.path.append("../src")
from models import *
from losses_and_regularizers import *
from denoise import Problem, ProblemImages, solve, initialize_experiment_report

is_debugging = False
if is_debugging:
    sys.argv += [
        "--subject",
        "1",
        "--noise_std",
        "0.15",
        "--fidelities",
        "Rician_Norm:1.0:0.15",
        "--regularizers",
        "Total_Variation:1.0:1.0:0.75",
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
        "--channels_list",
        "3",
        "128",
        "128",
        "128",
        "128",
        "--skip_sizes",
        "4",
        "4",
        "4",
        "4",
    ]

dataset_path = "../datasets/dataset"
results_path = "../results"
dev = (
    torch.device("cuda")
    if torch.cuda.is_available()  # and not is_debugging
    else torch.device("cpu")
)
psnr = PSNR()
ssim = SSIM(n_channels=1).to(dev)


# Function to generate a unique UUID not present in the target folder
def generate_unique_uuid(target_folder):
    import uuid
    import os

    while True:
        unique_id = str(uuid.uuid4())
        if unique_id not in os.listdir(
            target_folder
        ):  # Ensure it's not already present
            return unique_id


def save_best_img(best_img: Tensor, it: str, subject_idx: str, tag: str):
    print_image(
        (best_img.squeeze().cpu().detach().numpy() * 255).astype(np.uint8),
        f"{results_path}/im_{subject_idx}/def_denoised/{tag}{it}.png",
    )


def save_best_ssim(
    best_img: Tensor, ground_truth: Tensor, it: str, subject_idx: str, tag: str
):
    _, img_ssim = ssim_sk(
        ground_truth.squeeze().cpu().numpy(),
        best_img.squeeze().cpu().detach().numpy(),
        data_range=1.0,
        channel_axis=0,
        full=True,
    )  # img_ssim is HxW
    print_image(
        (img_ssim.clip(0, 1) * 255).astype(np.uint8),
        f"{results_path}/im_{subject_idx}/def_ssim/{tag}{it}.png",
    )


def get_dip_noise_fn(noise_type: Literal["", "Gaussian", "Rician"]):
    if noise_type == "Rician":
        return add_rician_noise
    if noise_type == "Gaussian":
        return add_gaussian_noise

    def id(x: Tensor):
        return x

    return id


def load_experiment_data(
    subject_idx: str,
    noise_std: str,
    print_noisy_image: bool = False,
):
    # cropping is no-op for our datasets
    gt_cpu = crop_image(
        load_serialized_image(
            f"{dataset_path}/im_{subject_idx}/gt.pt",
            normalize=False,
        )
    )
    mask_cpu = crop_image(
        load_gray_image(
            f"{dataset_path}/im_{subject_idx}/mask.png",
            as_mask=True,
        )
    )
    noisy_gt_cpu = crop_image(
        load_serialized_image(
            f"{dataset_path}/im_{subject_idx}/Std{noise_std}.pt",
            normalize=False,
        )  # image is contaminated (i.e., values not in the interval [0,1]), so we don't normalize
    )
    if print_noisy_image:
        print_image(
            (noisy_gt_cpu.numpy() * 255).astype(np.uint8),
            f"noisy_slice_subject_{subject_idx}_Std{noise_std}.png",
        )
    return noisy_gt_cpu, gt_cpu, mask_cpu


parser = argparse.ArgumentParser()
parser.add_argument(
    "--subject",
    type=str,
    help="The brain subject",
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
    "--dip_noise_type", choices=["Gaussian", "Rician", ""], default="Gaussian"
)
parser.add_argument("--max_its", type=int, default=15000)
parser.add_argument("--dip_noise_std", type=float, default=0.15)
parser.add_argument("--model", type=str, required=not is_debugging)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--N", type=int, default=1)
parser.add_argument(
    "--channels_list", type=int, nargs="*", default=[3, 128, 128, 128, 128, 128]
)
parser.add_argument("--skip_sizes", type=int, nargs="*", default=[4, 4, 4, 4, 4])


# ⌨️
args = parser.parse_args()
model = globals().get(args.model)

if model is None:
    print(f"Model {args.model} not found")
    quit(-1)
if model != UNet:
    model = model().to(dev)  # initialize model with default parameters
else:
    model = UNet(channels_list=args.channels_list, skip_sizes=args.skip_sizes).to(dev)
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

tag = generate_unique_uuid(f"{results_path}/im_{args.subject}/def_jsons/")

p: Problem = {
    "images": images,
    "psnr": psnr,
    "ssim": ssim,
    "max_its": args.max_its,
    "tag": tag,
    "optimizer": torch.optim.Adam(model.parameters(), lr=args.lr),
    "loss_config": composite_loss,
    "dip_config": {
        "noise_fn": get_dip_noise_fn(args.dip_noise_type),
        "std": args.dip_noise_std,
        "model": model,
        "seed": seed,
        "simultaneous_perturbations": args.N,
    },
    "image_name": f"im_{args.subject}",
}

report = initialize_experiment_report(p)

# for it in range(1, 3000):  # take 3000 'best images'
report, best_img = solve(p, report)
save_best_img(best_img, "", subject_idx=args.subject, tag=tag)
save_best_ssim(best_img, images.ground_truth, "", subject_idx=args.subject, tag=tag)
# if report["exit_code"] != 0:
#     break


with open(f"{results_path}/im_{args.subject}/def_jsons/{tag}.json", "w") as json_file:
    json.dump(report, json_file, indent=4)
