import torch
from torch import nn
import copy

from ..mlp import MLP
from ..basis import GaussianRandomFourierFeatures
from ..conv import MGNTransformerConv

# Typing
from torch import Tensor
from typing import List, Optional, Tuple


class Encoder(nn.Module):
    def __init__(self, init_node_dim, init_edge_dim, dim):
        super().__init__()
        self.init_node_dim = init_node_dim
        self.init_edge_dim = init_edge_dim
        self.dim = dim

        self.embed_node = nn.Sequential(MLP([init_node_dim, dim, dim], act=nn.SiLU()), nn.LayerNorm(dim))
        self.embed_edge = nn.Sequential(MLP([init_edge_dim, dim, dim], act=nn.SiLU()), nn.LayerNorm(dim))
    
    def forward(self, x:Tensor, edge_attr:Tensor) -> Tuple[Tensor, Tensor]:
        h_node = self.embed_node(x)
        h_edge = self.embed_edge(edge_attr)
        return h_node, h_edge


class Encoder_dpm(nn.Module):
    def __init__(self, init_node_dim, init_edge_dim, dim):
        super().__init__()
        self.init_node_dim = init_node_dim
        self.init_edge_dim = init_edge_dim
        self.dim = dim

        self.embed_node = nn.Sequential(MLP([init_node_dim, dim, dim], act=nn.SiLU()), nn.LayerNorm(dim))
        self.embed_edge = nn.Sequential(MLP([init_edge_dim, dim, dim], act=nn.SiLU()), nn.LayerNorm(dim))
        self.embed_time = nn.Sequential(
            GaussianRandomFourierFeatures(dim, input_dim=1),
            MLP([dim, dim, dim], act=nn.SiLU()),
            nn.LayerNorm(dim),
        )

    def forward(self, x:Tensor, edge_attr:Tensor, t:Tensor) -> Tuple[Tensor, Tensor]:
        # Encode nodes and edges
        h_node = self.embed_node(x)
        h_edge = self.embed_edge(edge_attr)
        
        # Add time embedding to node embedding
        h_node += self.embed_time(t)
        return h_node, h_edge


class Processor(nn.Module):
    def __init__(self, dim, ff_dim, num_convs, num_heads, cutoff):
        super().__init__()
        self.dim       = dim
        self.ff_dim    = ff_dim
        self.num_convs = num_convs
        self.num_heads = num_heads
        self.cutoff    = cutoff
        self.convs = nn.ModuleList(
            [copy.deepcopy(MGNTransformerConv(dim, ff_dim, num_heads, cutoff)) for _ in range(num_convs)]
        )

    def forward(self, h_node:Tensor, edge_index:Tensor, h_edge:Tensor, edge_len:Tensor) -> Tuple[Tensor, Tensor]:
        for conv in self.convs:
            h_node, h_edge = conv(h_node, edge_index, h_edge, edge_len)
        return h_node, h_edge


class Decoder(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.decoder = MLP([dim, dim, out_dim], act=nn.SiLU())
    
    def forward(self, h_node:Tensor) -> Tensor:
        return self.decoder(h_node)


class MGNTransformer(nn.Module):
    def __init__(self, encoder, processor, decoder):
        super().__init__()
        self.encoder   = encoder
        self.processor = processor
        self.decoder   = decoder
    
    def forward(self, x:Tensor, edge_index:Tensor, edge_attr:Tensor, edge_len:Tensor) -> Tensor:
        h_node, h_edge = self.encoder(x, edge_attr)
        h_node, h_edge = self.processor(h_node, edge_index, h_edge, edge_len)
        return self.decoder(h_node)
