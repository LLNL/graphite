import torch
from torch.nn  import Embedding, Linear, ModuleList, SiLU, Sequential, Sigmoid
from functools import partial
from ..basis   import bessel
from ..conv    import EGConv


class EGCONV_GNN(torch.nn.Module):
    """Edge-gated convolution GNN from https://arxiv.org/abs/2003.00982.
    """
    def __init__(self, dim=100, num_interactions=6, num_species=3, cutoff=6.0):
        super().__init__()

        self.dim              = dim
        self.num_interactions = num_interactions
        self.num_species      = num_species
        self.cutoff           = cutoff

        self.embed_atm        = Embedding(num_species, dim)
        self.embed_bnd        = partial(bessel, start=0, end=cutoff, num_basis=dim)

        self.atm_bnd_interactions = ModuleList()
        for _ in range(num_interactions):
            self.atm_bnd_interactions.append(EGConv(dim, dim))

        self.out = Sequential(
            Linear(dim, dim), SiLU(),
            Linear(dim, 1), Sigmoid(),
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.embed_atm.reset_parameters()
        for interaction in self.atm_bnd_interactions:
            interaction.reset_parameters()
        
    def forward(self, data):
        edge_index = data.edge_index
        h_atm      = self.embed_atm(data.x)
        h_bnd      = self.embed_bnd(data.edge_attr)

        for i in range(self.num_interactions):
            h_atm, h_bnd = self.atm_bnd_interactions[i](h_atm, edge_index, h_bnd)

        return self.out(h_atm)
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'dim={self.dim}, '
                f'num_interactions={self.num_interactions}, '
                f'num_species={self.num_species}, '
                f'cutoff={self.cutoff})')
