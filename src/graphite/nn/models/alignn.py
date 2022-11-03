import torch
from torch.nn       import Embedding, Linear, ModuleList, SiLU, Sequential
from torch_scatter  import scatter
from functools      import partial
from ..basis        import gaussian, bessel
from ..conv         import EGConv


class ALIGNN(torch.nn.Module):
    """ALIGNN model that uses auxiliary line graph to explicitly represent and encode bond angles.
    Reference: https://www.nature.com/articles/s41524-021-00650-1.
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
        
        self.head = Sequential(
            Linear(dim, dim), SiLU(),
        )
        
        self.out = Sequential(
            Linear(dim, 3),
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.embed_atm.reset_parameters()
        for i in range(self.num_interactions):
            self.atm_bnd_interactions[i].reset_parameters()
            self.bnd_ang_interactions[i].reset_parameters()

    def embed_ang(self, x_ang):
        cos_ang = torch.cos(x_ang)
        return gaussian(cos_ang, start=-1, end=1, num_basis=self.dim)

    def forward(self, data):
        edge_index_G = data.edge_index_G
        edge_index_A = data.edge_index_A
        h_atm        = self.embed_atm(data.x_atm)
        h_bnd        = self.embed_bnd(data.x_bnd)
        h_ang        = self.embed_ang(data.x_ang)
                
        for i in range(self.num_interactions):
            h_bnd, h_ang = self.bnd_ang_interactions[i](h_bnd, edge_index_A, h_ang)
            h_atm, h_bnd = self.atm_bnd_interactions[i](h_atm, edge_index_G, h_bnd)
        
        h = scatter(h_atm, data.x_atm_batch, dim=0, reduce='add')
        h = self.head(h)
        return self.out(h)
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'dim={self.dim}, '
                f'num_interactions={self.num_interactions}, '
                f'cutoff={self.cutoff})')
