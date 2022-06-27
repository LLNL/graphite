import torch
from torch.nn      import Embedding, Linear, ModuleList, SiLU, Sequential, Softplus
from torch_scatter import scatter
from functools     import partial
from ..basis       import gaussian, bessel
from ..conv        import EGConv


class ALIGNN_d_interpretable(torch.nn.Module):
    """Extended ALIGNN model that additionally incorporates dihedral angles.
    This version is designed to be "interpretable" by eventually summing all graph components,
    each of which holds a nonnegative scalar at the last layer,
    into a positive scalar quantity for final prediction.

    Reference: https://arxiv.org/abs/2109.11576.
    """
    def __init__(self, dim=100, num_interactions=6, num_species=3, cutoff=3.0):
        super().__init__()

        self.dim              = dim
        self.num_interactions = num_interactions
        self.cutoff           = cutoff

        self.embed_atm = Embedding(num_species, dim)
        self.embed_bnd = partial(bessel, start=0, end=cutoff, num_basis=dim)

        self.atm_bnd_interactions = ModuleList()
        self.bnd_ang_interactions = ModuleList()
        for _ in range(num_interactions):
            self.atm_bnd_interactions.append(EGConv(dim, dim))
            self.bnd_ang_interactions.append(EGConv(dim, dim))

        self.head_atm = Sequential(
            Linear(dim, 3), Softplus(),
        )
        self.head_bnd = Sequential(
            Linear(dim, 3), Softplus(),
        )
        self.head_ang = Sequential(
            Linear(dim, 3), Softplus(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.embed_atm.reset_parameters()
        for i in range(self.num_interactions):
            self.atm_bnd_interactions[i].reset_parameters()
            self.bnd_ang_interactions[i].reset_parameters()

    def embed_ang(self, x_ang, mask_dih_ang):
        cos_ang = torch.cos(x_ang)
        sin_ang = torch.sin(x_ang)

        h_ang = torch.zeros([len(x_ang), self.dim], device=x_ang.device)
        h_ang[~mask_dih_ang, :self.dim//2] = gaussian(cos_ang[~mask_dih_ang], start=-1, end=1, num_basis=self.dim//2)

        h_cos_ang = gaussian(cos_ang[mask_dih_ang], start=-1, end=1, num_basis=self.dim//4)
        h_sin_ang = gaussian(sin_ang[mask_dih_ang], start=-1, end=1, num_basis=self.dim//4)
        h_ang[mask_dih_ang, self.dim//2:] = torch.cat([h_cos_ang, h_sin_ang], dim=-1)

        return h_ang

    def forward(self, data):
        edge_index_G = data.edge_index_G
        edge_index_L = data.edge_index_L
        h_atm        = self.embed_atm(data.x_atm)
        h_bnd        = self.embed_bnd(data.x_bnd)
        h_ang        = self.embed_ang(data.x_ang, data.mask_dih_ang)

        for i in range(self.num_interactions):
            h_bnd, h_ang = self.bnd_ang_interactions[i](h_bnd, edge_index_L, h_ang)
            h_atm, h_bnd = self.atm_bnd_interactions[i](h_atm, edge_index_G, h_bnd)

        h_atm = self.head_atm(h_atm)
        h_bnd = self.head_bnd(h_bnd)
        h_ang = self.head_ang(h_ang)

        h_atm_agg = scatter(h_atm, data.x_atm_batch, dim=0, reduce='add')
        h_bnd_agg = scatter(h_bnd, data.x_bnd_batch, dim=0, reduce='add')
        h_ang_agg = scatter(h_ang, data.x_ang_batch, dim=0, reduce='add')

        out = h_atm_agg + h_bnd_agg + h_ang_agg
        return out, (h_atm, h_bnd, h_ang)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'dim={self.dim}, '
                f'num_interactions={self.num_interactions}, '
                f'cutoff={self.cutoff})')
