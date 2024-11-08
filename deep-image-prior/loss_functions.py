from torch import Tensor, i0, log, nn
from torch.special import i1


def gaussian_fidelity(prediction: Tensor, noisy_image: Tensor):
    return (prediction - noisy_image).square().mean()


def rician_fidelity(prediction: Tensor, noisy_image: Tensor, std: float):
    return (
        (prediction.square() / (2 * std**2))
        - log(i0((noisy_image.mul(prediction)) / std**2))
    ).mean()


def rician_norm_fidelity(prediction: Tensor, noisy_image: Tensor, std: float):
    # return (u*i0(u*f/(sigma**2)) - f*i1(u*f/(sigma**2))).square().mean() # EXPLOTA
    # return (u - f*(i1(u*f/(sigma**2))/i0(u*f/(sigma**2)))).square().mean() # INESTABLE

    return (
        (
            (
                prediction
                * (
                    i0(prediction * noisy_image / std**2)
                    / i1(prediction * noisy_image / std**2)
                )
                - noisy_image
            )
            / std**2
        )
        .square()
        .mean()
    )  # I0 < I1, así que I0/I1 parece que funciona


def rician_norm_fidelity_2(prediction: Tensor, noisy_image: Tensor, std: float):
    return (
        (
            (
                prediction
                - noisy_image
                * (
                    i1(prediction * noisy_image / std**2)
                    / i0(prediction * noisy_image / std**2)
                )
            )
            / (std**2 * prediction.shape[1] * prediction.shape[2])  # TODO: comprobar
        )
        .square()
        .mean()
    )
