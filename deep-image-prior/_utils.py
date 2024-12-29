from typing import Tuple
import torch
from torch import Tensor
import torchvision
import numpy as np
from numpy import ndarray
import torch.nn.functional as F


def __load_image__(path: str, mode: torchvision.io.ImageReadMode):
    return torchvision.io.read_image(path, mode) / 255


def load_rgb_image(path: str):
    return __load_image__(path, torchvision.io.image.ImageReadMode.RGB)


def load_gray_image(path: str, is_mask=False):
    img = __load_image__(path, torchvision.io.image.ImageReadMode.GRAY)
    if is_mask:
        img[img > 0] = img.max()
    return img


def load_serialized_image(path: str, is_mask=False, normalize=True) -> Tensor:
    img = torch.load(path, weights_only=False).to(dtype=torch.float32)
    if is_mask:
        img[img > 0] = img.max()
    if normalize:
        img = (img - img.min()) / (img.max() - img.min())
    return img


def get_bounding_box(mask: Tensor) -> tuple[tuple[int, int], tuple[int, int]]:
    indices = torch.nonzero(mask, as_tuple=True)
    top_left = (
        int(indices[-2].min().item()),
        int(indices[-1].min().item()),
    )  # (min_row, min_col)
    bottom_right = (
        int(indices[-2].max().item()),
        int(indices[-1].max().item()),
    )  # (max_row, max_col)
    return top_left, bottom_right


def crop_image(img, d=32):
    new_height = img.shape[-2] - img.shape[-2] % d
    new_width = img.shape[-1] - img.shape[-1] % d
    return img[..., :new_height, :new_width]


def pad_image(img: Tensor, new_height: int, new_width: int):
    channels = img.shape[0]
    padding = torch.zeros(channels, new_height, new_width)
    _, img_height, img_width = img.shape
    padding[:, :img_height, :img_width] = img
    return padding


def add_gaussian_noise(img: Tensor, std: float, avg: float = 0) -> Tensor:
    noise = avg + std * torch.randn(img.shape, device=img.device)
    return img + noise


# def add_rician_noise(img, std: float) -> torch.Tensor:
#     x_center, y_center = 0, 0
#     sample_size = img.shape[0] * img.shape[1] * img.shape[2]
#     distribution = np.random.normal(loc=[x_center, y_center], scale=std, size=(sample_size, 2))

#     noise_vector = np.linalg.norm(distribution, axis=1)

#     return img + noise_vector.reshape(img.shape).astype('float32')


def add_rician_noise_old(u: Tensor, sigma: float, kspace=False) -> torch.Tensor:
    if kspace:
        j = complex(0, 1)
        j = np.asarray(j)[None, None, None, ...]
        w, h, c = np.shape(u)
        omega = w * h * c
        scale = 1 / (np.pi)  # TODO justificar
        n = 1
        u_F = np.fft.fftn(u, s=[n * w, n * h], axes=(0, 1))
        f_F = u_F + np.sqrt(omega * scale) * (
            sigma * np.random.randn(n * w, n * h, c)
            + j * sigma * np.random.randn(n * w, n * h, c)
        )
        i_f_F = np.fft.ifftn(f_F, s=[n * w, n * h], axes=(0, 1))
        f = np.sqrt(np.real(i_f_F) ** 2 + np.imag(i_f_F) ** 2)
        # f = np.real(i_f_F)
        f = f[0:w, 0:h, :]
    else:
        f_real = u + sigma * np.random.randn(u.shape[0], u.shape[1], u.shape[2])
        f_imag = sigma * np.random.randn(u.shape[0], u.shape[1], u.shape[2])
        f = np.sqrt(f_real**2 + f_imag**2)
    return torch.tensor(f, dtype=torch.float32)


def add_rician_noise(img: Tensor, std: float) -> Tensor:
    img_real = img + std * torch.randn(img.shape, device=img.device)
    img_imag = std * torch.randn(img.shape, device=img.device)
    return (img_real**2 + img_imag**2).sqrt()


def print_image(img: ndarray, file_name: str | None):
    if file_name is None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        plt.close()
    else:
        from imageio import imwrite

        imwrite(uri=file_name, im=img)


def psnr_with_mask(
    img_1: Tensor | np.ndarray,
    img_2: Tensor | np.ndarray,
    mask: Tensor | np.ndarray,
    data_range=1.0,
):
    mask_size = (mask > 0).sum().item()
    mse = ((img_1 - img_2) ** 2 * mask).sum() / mask_size
    if (
        isinstance(img_1, Tensor)
        and isinstance(img_2, Tensor)
        and isinstance(mask, Tensor)
    ):
        return 10 * torch.log10(data_range**2 / mse)
    else:
        return 10 * np.log10(data_range**2 / mse)


def grads_old(image: Tensor, direction="forward") -> Tuple[Tensor, Tensor]:
    """Computes image gradients (dy/dx) for a given image."""
    channels, height, width = image.shape

    if direction == "forward":
        dy = image[..., 1:, :] - image[..., :-1, :]
        dx = image[..., :, 1:] - image[..., :, :-1]

        shape_y = [channels, 1, width]
        dy = torch.cat(
            [dy, torch.zeros(shape_y, device=image.device, dtype=image.dtype)], dim=1
        )
        dy = dy.view(image.shape)

        shape_x = [channels, height, 1]
        dx = torch.cat(
            [dx, torch.zeros(shape_x, device=image.device, dtype=image.dtype)], dim=2
        )
        dx = dx.view(image.shape)
    elif direction == "backward":
        dy = image[..., 1:, :] - image[..., :-1, :]
        dx = image[..., :, 1:] - image[..., :, :-1]

        shape_y = [channels, 1, width]
        dy = torch.cat(
            [torch.zeros(shape_y, device=image.device, dtype=image.dtype), dy], dim=1
        )
        dy = dy.view(image.shape)

        shape_x = [channels, height, 1]
        dx = torch.cat(
            [torch.zeros(shape_x, device=image.device, dtype=image.dtype), dx], dim=2
        )
        dx = dx.view(image.shape)
    else:
        raise ValueError("Invalid direction")

    return dy, dx


def laplacian(image: Tensor, p=1.0, eps=0.0):
    dx, dy = grads_old(image)
    grad_magnitude = (dx**2 + dy**2 + eps).sqrt().pow(p - 2)
    dx_weighted, _ = grads_old(dx * grad_magnitude, "backward")
    _, dy_weighted = grads_old(dy * grad_magnitude, "backward")
    return dx_weighted + dy_weighted


def roberts(image: Tensor):
    # Goal: detect edges in diagonal directions
    # Simple and fast, but sensitive to noise and will not detect horizontal/vertical edges
    f1 = F.conv2d(
        input=image,
        weight=torch.tensor([[1.0, 0.0], [0.0, -1.0]], device=image.device).reshape(
            1, 1, 2, 2
        ),
    )
    f2 = F.conv2d(
        input=image,
        weight=torch.tensor([[0.0, 1.0], [-1.0, 0.0]], device=image.device).reshape(
            1, 1, 2, 2
        ),
    )
    return f1, f2


def prewitt(image: Tensor):
    # Goal: detect edges in vertical and horizontal directions
    # Less sensitive to noise than Roberts, less precise than Sobel
    x_filter = torch.tensor(
        [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device=image.device
    ).reshape(1, 1, 3, 3)
    y_filter = torch.tensor(
        [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0], device=image.device
    ).reshape(1, 1, 3, 3)
    return F.conv2d(image, x_filter), F.conv2d(image, y_filter)


def sobel(image: Tensor):
    # Goal: detect edges in vertical and horizontal directions, emphasizing them
    # More robust to noise, and gives a good balance of noise reduction and edge detection
    x_filter = torch.tensor(
        [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0], device=image.device
    ).reshape(1, 1, 3, 3)
    y_filter = torch.tensor(
        [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0], device=image.device
    ).reshape(1, 1, 3, 3)
    conv_x = F.conv2d(input=image, weight=x_filter)
    conv_y = F.conv2d(input=image, weight=y_filter)
    return conv_x, conv_y


def kirsch(image: Tensor):
    f1 = torch.tensor(
        [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device=image.device
    ).reshape(1, 1, 3, 3)
    f2 = torch.tensor(
        [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0], device=image.device
    ).reshape(1, 1, 3, 3)
    f3 = torch.tensor(
        [-1.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0], device=image.device
    ).reshape(1, 1, 3, 3)
    f4 = torch.tensor(
        [0.0, 1.0, 1.0, -1.0, 0.0, 1.0, -1.0, -1.0, 0.0], device=image.device
    ).reshape(1, 1, 3, 3)
    filters = [f1, f2, f3, f4]
    return [F.conv2d(input=image, weight=f) for f in filters]


def dct_3(image: Tensor):
    filters = [
        torch.tensor(
            [-0.41, -0.41, -0.41, 0.0, 0.0, 0.0, 0.41, 0.41, 0.41], device=image.device
        ).reshape(1, 1, 3, 3),
        torch.tensor(
            [0.24, 0.24, 0.24, -0.47, -0.47, -0.47, 0.24, 0.24, 0.24],
            device=image.device,
        ).reshape(1, 1, 3, 3),
        torch.tensor(
            [0.41, 0.0, -0.41, 0.41, 0.0, -0.41, 0.41, 0.0, -0.41], device=image.device
        ).reshape(1, 1, 3, 3),
        torch.tensor(
            [-0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, -0.5], device=image.device
        ).reshape(1, 1, 3, 3),
        torch.tensor(
            [0.29, 0.0, -0.29, -0.58, 0.0, 0.58, 0.29, 0.0, -0.29], device=image.device
        ).reshape(1, 1, 3, 3),
        torch.tensor(
            [0.24, -0.47, 0.24, 0.24, -0.47, 0.24, 0.24, -0.47, 0.24],
            device=image.device,
        ).reshape(1, 1, 3, 3),
        torch.tensor(
            [-0.29, 0.58, -0.29, 0.0, 0.0, 0.0, 0.29, -0.58, 0.29], device=image.device
        ).reshape(1, 1, 3, 3),
        torch.tensor(
            [0.17, -0.33, 0.17, -0.33, 0.67, -0.33, 0.17, -0.33, 0.17],
            device=image.device,
        ).reshape(1, 1, 3, 3),
    ]
    return [F.conv2d(input=image, weight=f) for f in filters]


def grads(image: Tensor):
    if image.dim() == 3:
        image = image.unsqueeze(0)
        print(
            "grads - WARN: input image has 3 dimensions instead of 4. Adding new dim."
        )
    x_filter = torch.tensor(
        [0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], device=image.device
    ).reshape(1, 1, 3, 3)
    y_filter = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0], device=image.device
    ).reshape(1, 1, 3, 3)
    dx = F.conv2d(image, x_filter, padding="same")
    dy = F.conv2d(image, y_filter, padding="same")
    return dx, dy
