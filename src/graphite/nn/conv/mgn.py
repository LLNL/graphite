import torch
from torch import nn
from torch_geometric.utils import scatter

# Typing
from torch import Tensor
from typing import List, Optional, Tuple

from ..mlp import MLP


class EdgeProcessor(nn.Module):
    """Edge Processor for MeshGraphNets
    Args:
        hs (list of int): Input, hidden, and output dimensions of the MLP processor.
            Example: [64, 128, 128, 64].
    """
    def __init__(self, hs:List[int]):
        super().__init__()
        self.edge_mlp = nn.Sequential(MLP(hs=hs, act=nn.SiLU()), nn.LayerNorm(hs[-1]))

    def forward(self, x_i:Tensor, x_j:Tensor, edge_attr:Tensor) -> Tensor:
        out  = torch.cat([x_i, x_j, edge_attr], dim=-1)
        out  = self.edge_mlp(out)
        out += edge_attr
        return out


class NodeProcessor(nn.Module):
    """Node Processor for MeshGraphNets
    Args:
        hs (list of int): Input, hidden, and output dimensions of the MLP processor.
            Example: [64, 128, 128, 64].
    """
    def __init__(self, hs:List[int]):
        super().__init__()
        self.node_mlp = nn.Sequential(MLP(hs=hs, act=nn.SiLU()), nn.LayerNorm(hs[-1]))

    def forward(self, x:Tensor, edge_index:Tensor, edge_attr:Tensor) -> Tensor:
        j    = edge_index[1]
        out  = scatter(edge_attr, index=j, dim=0, dim_size=x.size(0))
        out  = torch.cat([x, out], dim=-1)
        out  = self.node_mlp(out)
        out += x
        return out


class MeshGraphNetsConv(nn.Module):
    """Graph convolution operation equivalent to MeshGraphNet's node/edge processors.

    Reference: https://arxiv.org/pdf/2010.03409v4.pdf

    Different from the original formulation, this version does not account for multiple edge sets.
    TODO: allow for multiple edge sets.
    """
    def __init__(self, node_dim:int, edge_dim:int):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.edge_processor = EdgeProcessor([node_dim*2 + edge_dim] + [edge_dim]*3)
        self.node_processor = NodeProcessor([node_dim   + edge_dim] + [node_dim]*3)
    
    def forward(self, x:Tensor, edge_index:Tensor, edge_attr:Tensor) -> Tuple[Tensor, Tensor]:
        i = edge_index[0]
        j = edge_index[1]
        edge_attr = self.edge_processor(x[i], x[j], edge_attr)
        x = self.node_processor(x, edge_index, edge_attr)
        return x, edge_attr

    def __repr__(self):
        return f'{self.__class__.__name__}(node_dim={self.node_dim}, edge_dim={self.edge_dim})'