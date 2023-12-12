import torch
from torch import nn
import copy

from ..mlp import MLP
from ..basis import GaussianRandomFourierFeatures
from ..conv import GraphormerConv, EquivariantVectorPrediction

# Typing
from torch import Tensor
from typing import List, Optional, Tuple


class Encoder(nn.Module):
    def __init__(self, init_node_dim, init_edge_dim, node_dim, edge_dim):
        super().__init__()
        self.init_node_dim = init_node_dim
        self.init_edge_dim = init_edge_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.embed_node = nn.Sequential(MLP([init_node_dim, node_dim, node_dim], act=nn.SiLU()), nn.LayerNorm(node_dim))
        self.embed_edge = nn.Sequential(MLP([init_edge_dim, edge_dim, edge_dim], act=nn.SiLU()), nn.LayerNorm(edge_dim))
    
    def forward(self, x:Tensor, edge_attr:Tensor) -> Tuple[Tensor, Tensor]:
        h_node = self.embed_node(x)
        h_edge = self.embed_edge(edge_attr)
        return h_node, h_edge


class Processor(nn.Module):
    def __init__(self, num_convs, node_dim, num_heads, ff_dim, edge_dim):
        super().__init__()
        self.num_convs = num_convs
        self.node_dim  = node_dim
        self.ff_dim    = ff_dim
        self.edge_dim  = edge_dim
        self.convs = nn.ModuleList(
            [copy.deepcopy(GraphormerConv(node_dim, num_heads, ff_dim, edge_dim)) for _ in range(num_convs)]
        )

    def forward(self, h_node:Tensor, edge_index:Tensor, edge_attr:Tensor) -> Tuple[Tensor, Tensor]:
        for conv in self.convs:
            h_node = conv(h_node, edge_index, edge_attr)
        return h_node


# class Decoder(nn.Module):
#     def __init__(self, dim, out_dim):
#         super().__init__()
#         self.dim = dim
#         self.out_dim = out_dim
#         self.decoder = MLP([dim, dim, out_dim], act=nn.SiLU())
    
#     def forward(self, h_node:Tensor) -> Tensor:
#         return self.decoder(h_node)


class Graphormer(nn.Module):
    def __init__(self, encoder, processor, decoder):
        super().__init__()
        self.encoder   = encoder
        self.processor = processor
        self.decoder   = decoder
    
    def forward(self, x:Tensor, edge_index:Tensor, edge_attr:Tensor) -> Tensor:
        h_node, h_edge = self.encoder(x, edge_attr)
        h_node         = self.processor(h_node, edge_index, h_edge)
        return self.decoder(h_node, edge_index, edge_attr)
