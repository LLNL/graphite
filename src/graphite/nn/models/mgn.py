import torch
from torch import nn

from graphite.nn import MLP
from ..conv import mgn_conv


class MeshGraphNet(torch.nn.Module):
    """MeshGraphNet in PyTorch-Geometric implementation.
    
    Reference:
    - https://arxiv.org/pdf/2010.03409v4.pdf

    Args:
        init_embed (function): Initial embedding function/class for nodes and edges.
        num_convs (int): Number of interaction/conv layers.
        node_dim (int): Number of features in node embedding throughout the model.
        edge_dim (int): Number of features in edge embedding throughout the model.
        out_dim (int): Number of features in node output.
    """
    def __init__(self,
        init_embed,
        num_convs  = 3,
        node_dim   = 64,
        edge_dim   = 64,
        out_dim    = 3,
    ):
        super().__init__()
        self.init_embed = init_embed
        self.num_convs  = num_convs
        self.node_dim   = node_dim
        self.edge_dim   = edge_dim

        self.convs = nn.ModuleList([mgn_conv(node_dim, edge_dim) for _ in range(num_convs)])
        self.out = MLP([node_dim, node_dim, out_dim], act=nn.SiLU())

    def forward(self, data):
        # Embedding
        data = self.init_embed(data)
        edge_index = data.edge_index
        h_node, h_edge = data.h_node, data.h_edge

        # Graph convolutions
        for conv in self.convs:
            h_node, h_edge, _ = conv(h_node, edge_index, h_edge)

        # Output layer
        return self.out(h)
