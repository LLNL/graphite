import torch
from torch_geometric.transforms import BaseTransform


class RattleParticles(BaseTransform):
    """Applies a random gaussian noise to paritcle positions. The standard deviation
    of the noise is drawn uniformly from a range [sigma_min, sigma_max].

    Args:
        sigma_min (float): The minimum standard deviation of the Gaussian noise.
        sigma_max (float): The maximum standard deviation of the Gaussian noise.
    """
    def __init__(self, sigma_max, sigma_min=0.001):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def __call__(self, data):
        # If `data` is a batch, apply noises of different magnitudes to the individual samples
        if data.batch is not None:
            sigma = torch.empty(data.num_graphs, device=data.pos.device).uniform_(self.sigma_min, self.sigma_max)
            sigma = sigma[data.batch, None]
        else:
            sigma = torch.empty(1, device=data.pos.device).uniform_(self.sigma_min, self.sigma_max)
        
        # Add noise
        eps       = torch.randn_like(data.pos)
        data.dx   = sigma*eps
        data.pos += data.dx

        # If `data` has edge vectors `edge_attr`, update them as well
        if data.edge_attr is not None:
            i, j = data.edge_index
            data.edge_attr += data.dx[j] - data.dx[i]

        # Potentially useful quantities to store
        data.sigma = sigma
        data.eps   = eps
        return data
    
    def __repr__(self):
        return f'{self.__class__.__name__}(sigma_min={self.sigma_min}, sigma_max={self.sigma_max})'
