from torch import Tensor, i0, log


def gaussian_fidelity(prediction: Tensor, noisy_image: Tensor):
    return (prediction - noisy_image).square().mean()


def rician_fidelity(prediction: Tensor, noisy_image: Tensor, std: float):
    return (
        (prediction.square() / (2 * std**2))
        - log(i0((noisy_image.mul(prediction)) / std**2))
    ).mean()
