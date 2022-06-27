import torch
from torch.nn      import Embedding, Linear, ModuleList, SiLU, Sequential, Sigmoid
from ..basis       import RadialBesselFunc
from ..conv        import EGConv


class EGCONV_GNN(torch.nn.Module):
    """Edge-gated convolution GNN from https://arxiv.org/abs/2003.00982.
    """
    def __init__(self, dim=100, num_interactions=6, num_species=3, cutoff=6.0):
        super().__init__()

        self.dim              = dim
        self.num_interactions = num_interactions
        self.num_species      = num_species
        self.cutoff           = cutoff

        self.embedding        = Embedding(num_species, dim)
        self.expand_bnd_dist  = RadialBesselFunc(dim, cutoff)

        self.atm_bnd_interactions = ModuleList()
        for _ in range(num_interactions):
            self.atm_bnd_interactions.append(EGConv(dim, dim))

        self.head = Sequential(
            Linear(dim, dim), SiLU(),
        )

        self.out = Sequential(
            Linear(dim, 1), Sigmoid(),
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.atm_bnd_interactions:
            interaction.reset_parameters()
        
    def forward(self, data):
        edge_index = data.edge_index
        h_atm      = self.embedding(data.x)
        h_bnd      = self.expand_bnd_dist(data.edge_attr)

        for i in range(self.num_interactions):
            h_atm, h_bnd = self.atm_bnd_interactions[i](h_atm, edge_index, h_bnd)

        h = self.head(h_atm)
        return self.out(h)
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'dim={self.dim}, '
                f'num_interactions={self.num_interactions}, '
                f'num_species={self.num_species}, '
                f'cutoff={self.cutoff})')
