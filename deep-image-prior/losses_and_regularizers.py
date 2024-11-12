from abc import abstractmethod
from typing import Protocol, TypedDict
import torch
from torch import Tensor, i0, nn, log
from torch.special import i1
from typing import List, Tuple
from _utils import grads, laplacian


# Base protocol for losses
class LossFunction(Protocol):
    @abstractmethod
    def loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        pass


# Base protocol for regularization terms
class RegularizationTerm(Protocol):
    @abstractmethod
    def regularization(self, prediction: Tensor) -> Tensor:
        pass


# Class for combining an abstract amount of losses & regularizers
class LossConfig(TypedDict):
    losses: List[Tuple[LossFunction, float]]
    regularizers: List[Tuple[RegularizationTerm, float]]


# Class for adding loss and regularization terms up to scaling factors
class CompositeLoss(nn.Module):
    def __init__(self, config: LossConfig):
        super(CompositeLoss, self).__init__()
        self.config = config

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        total_loss = torch.tensor(0.0, device=prediction.device)

        # Aggregate multiple losses
        for loss_fn, lambda_loss in self.config["losses"]:
            total_loss += lambda_loss * loss_fn.loss(prediction, target)

        # Aggregate multiple regularizers
        for reg_fn, lambda_reg in self.config["regularizers"]:
            total_loss += lambda_reg * reg_fn.regularization(prediction)

        return total_loss


class Gaussian(LossFunction):
    def loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        return torch.nn.functional.mse_loss(prediction, target)


class Rician(LossFunction):
    def __init__(self, std: float):
        self.std = std

    def loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        return (
            prediction.square() / (2 * self.std**2)
            - log(i0(target * prediction / self.std**2))
        ).mean()


class Rician_Norm(LossFunction):
    def __init__(self, std: float):
        self.std = std

    def loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        r_inv = i0(prediction * target / self.std**2) / i1(
            prediction * target / self.std**2
        )
        return (
            (
                (prediction * r_inv - target)
                / (self.std**2 * prediction.shape[1] * prediction.shape[2])
            )
            .square()
            .mean()
        )


class Laplacian_Rician_Norm(LossFunction):
    def __init__(self, σ: float, λ: float):
        self.σ = σ
        self.λ = λ

    def loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        r_inv = i0(prediction * target / self.σ**2) / i1(
            prediction * target / self.σ**2
        )
        Δ = laplacian(prediction, p=1.0, eps=1e-6)
        return (
            (
                (Δ * r_inv - self.λ * (prediction * r_inv - target))
                / (self.σ**2 * prediction.shape[1] * prediction.shape[2])
            )
            .square()
            .mean()
        )


class TotalVariation(RegularizationTerm):
    def __init__(self, p=1.0, eps=1e-6):
        self.p = p
        self.eps = eps

    def regularization(self, prediction: Tensor) -> Tensor:
        grad_x, grad_y = grads(prediction)
        return (grad_x**2 + grad_y**2 + self.eps).pow(self.p / 2).mean()
