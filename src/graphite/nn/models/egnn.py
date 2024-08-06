import torch
from torch import nn
import copy

from ..mlp import MLP
from ..convs import EGNNConv
from ..basis import GaussianRandomFourierFeatures

# Typing
from torch import Tensor
from typing import List, Optional, Tuple


class Encoder(nn.Module):
    def __init__(self, init_node_dim: int, init_edge_dim: int, node_dim: int, edge_dim: Optional[int] = None) -> None:
        super().__init__()
        self.init_node_dim = init_node_dim
        self.init_edge_dim = init_edge_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.embed_node = MLP([init_node_dim, node_dim, node_dim], act=nn.SiLU())
        if edge_dim is not None:
            self.embed_edge = MLP([init_edge_dim, edge_dim, edge_dim], act=nn.SiLU())
        else:
            self.embed_edge = lambda x: x

    def forward(self, x: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        h_node = self.embed_node(x)
        h_edge = self.embed_edge(edge_attr)
        return h_node, h_edge


class Processor(nn.Module):
    def __init__(self, num_convs: int, dim: int) -> None:
        super().__init__()
        self.num_convs = num_convs
        self.dim       = dim
        self.convs = nn.ModuleList(
            [copy.deepcopy(EGNNConv(dim)) for _ in range(num_convs)]
        )

    def forward(self, h: Tensor, pos: Tensor, edge_index: Tensor, a_ij: Tensor) -> Tuple[Tensor, Tensor]:
        for conv in self.convs:
            h, pos = conv(h, pos, edge_index, a_ij)
        return h, pos


class Decoder(nn.Module):
    def __init__(self, node_dim: int, out_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.decoder = MLP([node_dim, node_dim, out_dim], act=nn.SiLU())

    def forward(self, h_node: Tensor) -> Tensor:
        return self.decoder(h_node)


class EGNN(nn.Module):
    def __init__(self, encoder, processor, decoder):
        super().__init__()
        self.encoder   = encoder
        self.processor = processor
        self.decoder   = decoder

    def forward(self, x: Tensor, pos: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        h_node, h_edge = self.encoder(x, edge_attr)
        h_node, pos    = self.processor(h_node, pos, edge_index, h_edge)
        return self.decoder(h_node), pos
