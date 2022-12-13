import torch
from torch_geometric.transforms import BaseTransform


class RattleParticles(BaseTransform):
    """Applies a random gaussian noise to paritcle positions. The standard deviation
    of the noise is drawn uniformly from a range [sigma_min, sigma_max].

    Args:
        sigma_min (float): The minimum standard deviation of the Gaussian noise.
        sigma_max (float): The maximum standard deviation of the Gaussian noise.
    """
    def __init__(self,
        sigma_min = 0.0,
        sigma_max = 0.3,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def __call__(self, data):
        sigma = torch.empty(1, device=data.pos.device).uniform_(self.sigma_min, self.sigma_max)
        eps   = torch.randn_like(data.pos)

        data.disp  = sigma*eps
        data.pos  += data.disp
        data.sigma = sigma
        data.eps   = eps
        return data
    
    def __repr__(self):
        return f'{self.__class__.__name__}(sigma_min={self.sigma_min}, sigma_max={self.sigma_max})'
