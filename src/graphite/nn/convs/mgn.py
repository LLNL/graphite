import torch
from torch import nn
from torch_geometric.utils import scatter

from ..mlp import MLP

# Typing
from torch import Tensor
from typing import List, Optional, Tuple


class EdgeProcessor(nn.Module):
    def __init__(self, dims: List[int]) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(MLP(dims, act=nn.SiLU()), nn.LayerNorm(dims[-1]))

    def forward(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        out = torch.cat([x_i, x_j, edge_attr], dim=-1)
        out = self.edge_mlp(out)
        return edge_attr + out


class NodeProcessor(nn.Module):
    def __init__(self, dims: List[int]) -> None:
        super().__init__()
        self.node_mlp = nn.Sequential(MLP(dims, act=nn.SiLU()), nn.LayerNorm(dims[-1]))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        j   = edge_index[1]
        out = scatter(edge_attr, index=j, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=-1)
        out = self.node_mlp(out)
        return x + out


class MeshGraphNetsConv(nn.Module):
    """MeshGraphNets convolution/processor operation.

    Reference: https://arxiv.org/pdf/2010.03409v4.pdf

    Notes:
        Different from the original formulation, this version does not account for multiple edge sets.
        TODO: allow for multiple edge sets.
    """
    def __init__(self, node_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.edge_processor = EdgeProcessor([node_dim*2 + edge_dim] + [edge_dim]*3)
        self.node_processor = NodeProcessor([node_dim   + edge_dim] + [node_dim]*3)
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        i = edge_index[0]
        j = edge_index[1]
        edge_attr = self.edge_processor(x[i], x[j], edge_attr)
        x = self.node_processor(x, edge_index, edge_attr)
        return x, edge_attr

    def extra_repr(self) -> str:
        return f'node_dim={self.node_dim}, edge_dim={self.edge_dim}'