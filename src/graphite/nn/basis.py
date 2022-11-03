import torch


def bessel(x, start=0.0, end=1.0, num_basis=8, eps=1e-5):
    """Expand scalar features into (radial) Bessel basis function values.
    """
    x = x[..., None] - start + eps
    c = end - start
    n = torch.arange(1, num_basis+1, dtype=x.dtype, device=x.device)
    return ((2/c)**0.5) * torch.sin(n*torch.pi*x / c) / x


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


class GaussianFourierProjection(torch.nn.Module):
    """Gaussian random features from the paper titled
    'Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains'.
    Reference: https://arxiv.org/abs/2006.10739
    """
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = torch.nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)