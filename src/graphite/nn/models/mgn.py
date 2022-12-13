import torch
from torch import nn

from graphite.nn import MLP
from ..conv import mgn_conv


class MeshGraphNet(torch.nn.Module):
    """MeshGraphNet in PyTorch-Geometric implementation.
    
    Reference:
    - https://arxiv.org/pdf/2010.03409v4.pdf
    """
    def __init__(self, num_convs=3, node_dim=64, edge_dim=64, out_dim=3):
        super().__init__()
        self.num_convs = num_convs
        self.node_dim  = node_dim
        self.edge_dim  = edge_dim

        self.convs = nn.ModuleList([mgn_conv(node_dim, edge_dim) for _ in range(num_convs)])
        self.out = MLP([node_dim]*3 + [out_dim])

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for conv in self.convs:
            x, edge_attr, _ = conv(x, edge_index, edge_attr)

        return self.out(h)
