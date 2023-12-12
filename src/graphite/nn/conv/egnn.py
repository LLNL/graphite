import torch
from torch import nn
from torch_geometric.utils import scatter

# Typing
from torch import Tensor
from typing import List, Optional, Tuple

from ..mlp import MLP


class EGNNConv(nn.Module):
    """EGNN's equivariant graph convolution.

    References:
    - https://arxiv.org/pdf/2102.09844.pdf
    - https://arxiv.org/pdf/2203.17003.pdf
    """
    def __init__(self, dim:int, a_dim:int = 1):
        super().__init__()
        self.node_dim = dim
        self.phi_e   = nn.Sequential(MLP([dim*2 + 1 + a_dim, dim, dim], act=nn.SiLU()), nn.SiLU())
        self.phi_x   =               MLP([dim*2 + 1 + a_dim, dim,   1], act=nn.SiLU())
        self.phi_h   =               MLP([dim*2,             dim, dim], act=nn.SiLU())
        self.phi_inf = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, h:Tensor, x:Tensor, edge_index:Tensor, a_ij:Tensor) -> Tuple[Tensor, Tensor]:
        i = edge_index[0]
        j = edge_index[1]
        hhda = torch.cat([h[i], h[j], torch.linalg.norm(x[i]-x[j], dim=-1, keepdim=True), a_ij], dim=-1)
        m_ij = self.phi_e(hhda)
        x    = x + scatter(
            (x[i] - x[j])*self.phi_x(hhda),
            index=j, dim=0, dim_size=x.size(0), reduce='mean',
        )
        m    = scatter(self.phi_inf(m_ij)*m_ij, index=j, dim=0, dim_size=x.size(0))
        h    = h + self.phi_h(torch.cat([h, m], dim=-1))
        return h, x

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim})'