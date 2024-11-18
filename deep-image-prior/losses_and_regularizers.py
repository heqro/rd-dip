from abc import abstractmethod
from typing import Protocol, TypedDict
import torch
from torch import Tensor, i0, nn, log
from torch.special import i1
from typing import List, Tuple
from _utils import grads, laplacian
from torch.nn import functional as F


# Base protocol for losses
class LossFunction(Protocol):
    @abstractmethod
    def loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        pass

    @abstractmethod
    def get_mask() -> Tensor:
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
    def __init__(self, use_mask=False):
        self.use_mask = use_mask
        self.mask = None

    def loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        if self.mask is None:  # initialize mask for the first time
            self.mask = torch.ones_like(
                prediction, requires_grad=self.use_mask
            )  # learnable mask only if necessary
        if (self.mask < 0).any():
            print(f"WARN: {self.__class__.__name__} mask features negative entries")
        return (self.mask * (prediction - target)).square().mean()

    def get_mask(self) -> Tensor:
        if self.mask is None:
            raise Exception("Mask is None. Apply loss() to initialize it.")
        return self.mask


class Rician(LossFunction):
    def __init__(self, std: float):
        self.std = std

    def loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        return (
            prediction.square() / (2 * self.std**2)
            - log(i0(target * prediction / self.std**2))
        ).mean()

    def get_mask(self):
        raise NotImplementedError("Not implemented yet")


class Rician_Norm_Unstable(LossFunction):
    def __init__(self, std: float):
        self.std = std

    def loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        r = i1(prediction * target / self.std**2) / i0(
            prediction * target / self.std**2
        )
        return (
            (
                (prediction - r * target)
                / (self.std**2 * prediction.shape[1] * prediction.shape[2])
            )
            .square()
            .mean()
        )

    def get_mask(self):
        raise NotImplementedError("Not implemented yet")


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

    def get_mask(self):
        raise NotImplementedError("Not implemented yet")


class Laplacian_Rician_Norm(LossFunction):
    def __init__(self, σ: float, λ: float, p: float = 1.0):
        self.σ = σ
        self.λ = λ
        self.p = p

    def loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        r_inv = i0(prediction * target / self.σ**2) / i1(
            prediction * target / self.σ**2
        )
        Δ = laplacian(prediction, p=self.p, eps=1e-6)
        return (
            (
                (Δ - self.λ * (prediction * r_inv - target))
                / (self.σ**2 * prediction.shape[1] * prediction.shape[2])
            )
            .square()
            .mean()
        )

    def get_mask(self):
        raise NotImplementedError("Not implemented yet")


class Total_Variation(RegularizationTerm):
    def __init__(self, p=1.0, ε=1e-6):
        self.p = p
        self.ε = ε

    def regularization(self, prediction: Tensor) -> Tensor:
        grad_x, grad_y = grads(prediction)
        return (grad_x**2 + grad_y**2 + self.ε).pow(self.p / 2).mean()


class Discrete_Cosine_Transform(RegularizationTerm):
    def __init__(
        self, device: torch.device, dim: int = 3, p: float = 1.0, ε: float = 1e-6
    ):

        def generate_dct_filters(n: int, skip_first=True):
            from scipy.fft import idct

            filters = torch.zeros(size=(n**2, n, n))

            for i in range(n):
                for j in range(n):
                    filter = torch.zeros((n, n))
                    filter[i, j] = 1
                    filters[i * n + j, ...] = torch.from_numpy(
                        idct(idct(filter, norm="ortho").T, norm="ortho")
                    )
            return filters[skip_first:]

        self.p, self.ε = p, ε
        self.filters = nn.Parameter(
            generate_dct_filters(dim)[:, None, ...], requires_grad=False
        ).to(device)

    def phi(self, x: Tensor, p=1.0):
        return (1 / p) * ((x**2).sum(dim=1) + self.ε**2).pow(p / 2)

    def regularization(self, prediction: Tensor) -> Tensor:
        if len(prediction.shape) == 3:
            prediction = prediction[:, None, ...]
        if len(prediction.shape) != 4:
            raise Exception(
                f"Expected prediction to have shape 4, but it is {len(prediction.shape)}"
            )
        b, c, w, h = prediction.shape
        prediction = prediction.view(c, b, w, h)  # more efficient
        # z = z.permute(1, 0, 2, 3) #
        k_x_3 = F.conv2d(prediction, weight=self.filters, groups=1, dilation=1)
        y = (
            self.phi(k_x_3, p=self.p) / self.filters.shape[0]
        )  # divide by number of filters

        return y.mean()
