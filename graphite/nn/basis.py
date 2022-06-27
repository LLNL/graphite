import torch
import math


__all__ = ['scalar2basis']


def bessel(x, start=0.0, end=1.0, num_basis=8, eps=1e-5):
    """Expand scalar features into (radial) Bessel basis function values.
    """
    x = x[..., None] - start + eps
    c = end - start
    n = torch.arange(1, num_basis+1, dtype=x.dtype, device=x.device)
    return math.sqrt(2/c) * torch.sin(n*math.pi*x / c) / x


def gaussian(x, start=0.0, end=1.0, num_basis=8):
    """Expand scalar features into Gaussian basis function values.
    """
    mu = torch.linspace(start, end, num_basis, dtype=x.dtype, device=x.device)
    step = mu[1] - mu[0]
    diff = (x[..., None] - mu) / step
    return diff.pow(2).neg().exp().div(1.12) # division by 1.12 so that sum of square is roughly 1


def scalar2basis(x, start, end, num_basis, basis='gaussian'):
    """Expand scalar features into basis function values.
    Reference: https://docs.e3nn.org/en/stable/api/math/math.html#e3nn.math.soft_one_hot_linspace.
    """
    funcs = {
        'gaussian': gaussian,
        'bessel': bessel,
    }
    return funcs[basis](x, start, end, num_basis)


# Older code
class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0, stop=15, num_gaussians=150, gamma=1):
        super().__init__()
        mu = torch.linspace(start, stop, num_gaussians)
        self.gamma = gamma
        self.register_buffer('mu', mu)
        
    def forward(self, dist):
        dist_offset = dist.view(-1, 1) - self.mu.view(1, -1)
        return torch.exp(-self.gamma * torch.pow(dist_offset, 2))


class RadialBesselFunc(torch.nn.Module):
    def __init__(self, num_radial=16, cutoff=3.5):
        super().__init__()
        n = torch.arange(1, num_radial+1)
        c = torch.tensor(cutoff)
        self.register_buffer('n', n)
        self.register_buffer('c', c)

    def forward(self, d):
        return torch.sqrt(2/self.c) * torch.sin(self.n*math.pi*d / self.c) / d
