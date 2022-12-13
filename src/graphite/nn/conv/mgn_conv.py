import torch
from torch              import nn
from torch_scatter      import scatter
from torch_geometric.nn import MetaLayer

from graphite.nn import MLP


class EdgeProcessor(nn.Module):
    """Edge Processor for MeshGraphNet
    Args:
        hs (list of int): Input, hidden, and output dimensions of the MLP processor.
    """
    def __init__(self, hs):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            MLP(hs=hs, act=nn.SiLU()),
            nn.LayerNorm(hs[-1]),
        )

    def forward(self, x_i, x_j, edge_attr, u=None, batch=None):
        out  = torch.cat([x_i, x_j, edge_attr], dim=-1)
        out  = self.edge_mlp(out)
        out += edge_attr
        return out


class NodeProcessor(nn.Module):
    """Node Processor for MeshGraphNet
    Args:
        hs (list of int): Input, hidden, and output dimensions of the MLP processor.
    """
    def __init__(self, hs):
        super().__init__()
        self.node_mlp = nn.Sequential(
            MLP(hs=hs, act=nn.SiLU()),
            nn.LayerNorm(hs[-1]),
        )

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        i, j = edge_index
        out  = scatter(edge_attr, index=i, dim=0)
        out  = torch.cat([x, out], dim=-1)
        out  = self.node_mlp(out)
        out += x
        return out


def mgn_conv(node_dim, edge_dim):
    """A graph convolution operation equivalent to MeshGraphNet's node/edge processors.

    Reference:
    - https://arxiv.org/pdf/2010.03409v4.pdf
    """
    edge_model = EdgeProcessor(hs=[node_dim*2+edge_dim]+[edge_dim]*3)
    node_model = NodeProcessor(hs=[node_dim+edge_dim]+[node_dim]*3)
    return MetaLayer(edge_model=edge_model, node_model=node_model)