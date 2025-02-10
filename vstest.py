import _utils
import torch
import numpy as np
import matplotlib.pyplot as plt


def load_experiment_data(
    subject_idx: str,
    noise_std: str,
    print_noisy_image: bool = False,
):
    gt_cpu = _utils.crop_image(
        _utils.load_serialized_image(
            f".brainweb_test_data/im_{subject_idx}/gt.pt",
            normalize=False,
        )
    )  # image is already normalized
    mask_cpu = _utils.crop_image(
        _utils.load_gray_image(
            f".brainweb_test_data/im_{subject_idx}/mask.png",
            is_mask=True,
        )
    )
    noisy_gt_cpu = _utils.crop_image(
        _utils.load_serialized_image(
            f".brainweb_test_data/im_{subject_idx}/Std{noise_std}.pt",
            normalize=False,
        )  # image is contaminated (i.e., values not in the interval [0,1]), so we don't normalize
    )
    if print_noisy_image:
        _utils.print_image(
            (noisy_gt_cpu.numpy() * 255).astype(np.uint8),
            f"noisy_slice_subject_{subject_idx}_Std{noise_std}.png",
        )
    return noisy_gt_cpu, gt_cpu, mask_cpu


sigma = "0.20"
noisy, gt, mask = load_experiment_data("1", sigma, False)


def anscombe_transform(x: torch.Tensor, sigma: float):
    return torch.sqrt(torch.clip(x**2 - sigma**2, 0))


gaussian_noisy = anscombe_transform(noisy, float(sigma))
_utils.print_image((gaussian_noisy.numpy()[0] * 255).astype(np.uint8), "Gnoisy.png")
_utils.print_image((noisy.numpy()[0] * 255).astype(np.uint8), "Rnoisy.png")

gaussian_noisy_list = gaussian_noisy.numpy().flatten()

plt.hist(
    gaussian_noisy_list[gaussian_noisy_list > 0],
    bins=50,
    color="blue",
    alpha=0.7,
    edgecolor="black",
)
plt.title("1D Histogram of Image Pixel Values")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.savefig("GHist.pdf", bbox_inches="tight")
plt.close()

rician_noisy_list = noisy.numpy().flatten()

plt.hist(
    rician_noisy_list[rician_noisy_list > 0],
    bins=50,
    color="blue",
    alpha=0.7,
    edgecolor="black",
)
plt.title("1D Histogram of Image Pixel Values")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.savefig("RHist.pdf", bbox_inches="tight")
plt.close()
