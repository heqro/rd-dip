from typing import Tuple
import torch
from torch import Tensor
import torchvision
import numpy as np
from numpy import ndarray


def __load_image__(path: str, mode: torchvision.io.ImageReadMode):
    return torchvision.io.read_image(path, mode) / 255


def load_rgb_image(path: str):
    return __load_image__(path, torchvision.io.image.ImageReadMode.RGB)


def load_gray_image(path: str, is_mask=False):
    img = __load_image__(path, torchvision.io.image.ImageReadMode.GRAY)
    if is_mask:
        img[img > 0] = img.max()
    return img


def crop_image(img, d=32):
    new_height = img.shape[0] - img.shape[0] % d
    new_width = img.shape[1] - img.shape[1] % d
    return img[:, :new_height, :new_width]


def pad_image(img: Tensor, new_height: int, new_width: int):
    channels = img.shape[0]
    padding = torch.zeros(channels, new_height, new_width)
    _, img_height, img_width = img.shape
    padding[:, :img_height, :img_width] = img
    return padding


def add_gaussian_noise(img: Tensor, avg: float, std: float) -> Tensor:
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
    img_real = img + std * torch.randn(img.shape)
    img_imag = std * torch.randn(img.shape)
    return (img_real**2 + img_imag**2).sqrt().to(img.device)


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
    return 10 * torch.log10(data_range**2 / mse)


def grads(image: Tensor, direction="forward") -> Tuple[Tensor, Tensor]:
    """Computes image gradients (dy/dx) for a given image."""
    batch_size, channels, height, width = image.shape

    if direction == "forward":
        dy = image[..., 1:, :] - image[..., :-1, :]
        dx = image[..., :, 1:] - image[..., :, :-1]

        shape_y = [batch_size, channels, 1, width]
        dy = torch.cat(
            [dy, torch.zeros(shape_y, device=image.device, dtype=image.dtype)], dim=2
        )
        dy = dy.view(image.shape)

        shape_x = [batch_size, channels, height, 1]
        dx = torch.cat(
            [dx, torch.zeros(shape_x, device=image.device, dtype=image.dtype)], dim=3
        )
        dx = dx.view(image.shape)
    elif direction == "backward":
        dy = image[..., 1:, :] - image[..., :-1, :]
        dx = image[..., :, 1:] - image[..., :, :-1]

        shape_y = [batch_size, channels, 1, width]
        dy = torch.cat(
            [torch.zeros(shape_y, device=image.device, dtype=image.dtype), dy], dim=2
        )
        dy = dy.view(image.shape)

        shape_x = [batch_size, channels, height, 1]
        dx = torch.cat(
            [torch.zeros(shape_x, device=image.device, dtype=image.dtype), dx], dim=3
        )
        dx = dx.view(image.shape)
    else:
        raise ValueError("Invalid direction")

    return dy, dx
