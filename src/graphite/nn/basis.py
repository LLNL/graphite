import torch
from torch import nn

# Typing
from torch import Tensor
from typing import List, Optional, Tuple


def bessel(x: Tensor, start: float = 0.0, end: float = 1.0, num_basis: int = 8, eps: float = 1e-5) -> Tensor:
    """Expand scalar features into (radial) Bessel basis function values.
    """
    x = x[..., None] - start + eps
    c = end - start
    n = torch.arange(1, num_basis+1, dtype=x.dtype, device=x.device)
    return ((2/c)**0.5) * torch.sin(n*torch.pi*x / c) / x


def gaussian(x: Tensor, start: float = 0.0, end: float = 1.0, num_basis: int = 8) -> Tensor:
    """Expand scalar features into Gaussian basis function values.
    """
    mu = torch.linspace(start, end, num_basis, dtype=x.dtype, device=x.device)
    step = mu[1] - mu[0]
    diff = (x[..., None] - mu) / step
    return diff.pow(2).neg().exp().div(1.12) # division by 1.12 so that sum of square is roughly 1


def scalar2basis(x: Tensor, start: float, end: float, num_basis: int, basis: str = 'gaussian'):
    """Expand scalar features into basis function values.
    Reference: https://docs.e3nn.org/en/stable/api/math/math.html#e3nn.math.soft_one_hot_linspace.
    """
    funcs = {
        'gaussian': gaussian,
        'bessel': bessel,
    }
    return funcs[basis](x, start, end, num_basis)


class Bessel(nn.Module):
    def __init__(self, start: float = 0.0, end: float = 1.0, num_basis: int = 8, eps: float = 1e-5) -> None:
        super().__init__()
        self.start     = start
        self.end       = end
        self.num_basis = num_basis
        self.eps       = eps
        self.register_buffer('n', torch.arange(1, num_basis+1, dtype=torch.float))

    def forward(self, x: Tensor) -> Tensor:
        x = x[..., None] - self.start + self.eps
        c = self.end - self.start
        return ((2/c)**0.5) * torch.sin(self.n*torch.pi*x / c) / x

    def extra_repr(self) -> str:
        return f'start={self.start}, end={self.end}, num_basis={self.num_basis}, eps={self.eps}'


class Gaussian(nn.Module):
    def __init__(self, start: float = 0.0, end: float = 1.0, num_basis: int = 8) -> None:
        super().__init__()
        self.start     = start
        self.end       = end
        self.num_basis = num_basis
        self.register_buffer('mu', torch.linspace(start, end, num_basis))
    
    def forward(self, x: Tensor) -> Tensor:
        step = self.mu[1] - self.mu[0]
        diff = (x[..., None] - self.mu) / step
        return diff.pow(2).neg().exp().div(1.12) # division by 1.12 so that sum of square is roughly 1

    def extra_repr(self) -> str:
        return f'start={self.start}, end={self.end}, num_basis={self.num_basis}'


class GaussianRandomFourierFeatures(nn.Module):
    """Gaussian random Fourier features.

    Reference: https://arxiv.org/abs/2006.10739
    """
    def __init__(self, embed_dim: int, input_dim: int = 1, sigma: float = 1.0) -> None:
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.register_buffer('B', torch.randn(input_dim, embed_dim//2) * sigma)

    def forward(self, v: Tensor) -> Tensor:
        v_proj =  2 * torch.pi * v @ self.B
        return torch.cat([torch.cos(v_proj), torch.sin(v_proj)], dim=-1)
