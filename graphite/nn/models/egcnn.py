import torch
from torch.nn import Embedding, Linear, ModuleList, SiLU, Sequential
from torch_scatter import scatter
from functools import partial
from ..basis import bessel
from ..conv  import EGConv


class EGCNN(torch.nn.Module):
    """Edge-gated convolution GNN.
    This version is intended for spectroscopy prediction of Cu aqua complexes.
    """
    def __init__(self, dim=100, num_interactions=6, num_species=3, cutoff=6.0):
        super(EGCNN, self).__init__()

        self.dim              = dim
        self.num_interactions = num_interactions
        self.cutoff           = cutoff

        self.embed_atm = Embedding(num_species, dim)
        self.embed_bnd = partial(bessel, start=0, end=cutoff, num_basis=dim)

        self.atm_bnd_interactions = ModuleList([EGConv(dim, dim) for _ in range(num_interactions)])

        self.head = Sequential(
            Linear(dim, dim), SiLU(),
        )

        self.out = Sequential(
            Linear(dim, 3),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.embed_atm.reset_parameters()
        for interaction in self.atm_bnd_interactions:
            interaction.reset_parameters()

    def forward(self, data):
        edge_index = data.edge_index_G
        h_atm      = self.embed_atm(data.x_atm)
        h_bnd      = self.embed_bnd(data.x_bnd)

        for conv in self.atm_bnd_interactions:
            h_atm, h_bnd = conv(h_atm, edge_index, h_bnd)

        h = scatter(h_atm, data.x_atm_batch, dim=0, reduce='add')
        h = self.head(h)
        return self.out(h), h_atm

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'dim={self.dim}, '
                f'num_interactions={self.num_interactions}, '
                f'cutoff={self.cutoff})')
