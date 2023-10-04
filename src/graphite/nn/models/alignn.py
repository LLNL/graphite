import torch
from torch import nn
from functools import partial
import copy

from ..mlp import MLP
from ..basis import gaussian, bessel
from ..conv import MeshGraphNetsConv

# Typing
from torch import Tensor
from typing import List, Optional, Tuple


class Encoder(nn.Module):
    """ALIGNN/ALIGNN-d Encoder.
    """
    def __init__(self, num_species, init_bnd_dim=4, init_ang_dim=4, dim=128):
        super().__init__()
        self.num_species  = num_species
        self.init_bnd_dim = init_bnd_dim
        self.init_ang_dim = init_ang_dim
        self.dim          = dim
        
        self.embed_atm = nn.Sequential(MLP([num_species, dim, dim],  act=nn.SiLU()), nn.LayerNorm(dim))
        self.embed_bnd = nn.Sequential(MLP([init_bnd_dim, dim, dim], act=nn.SiLU()), nn.LayerNorm(dim))
        self.embed_ang = nn.Sequential(MLP([init_ang_dim, dim, dim], act=nn.SiLU()), nn.LayerNorm(dim))

    def forward(self,  x_atm:Tensor, x_bnd:Tensor, x_ang:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        h_atm = self.embed_atm(x_atm)
        h_bnd = self.embed_bnd(x_bnd)
        h_ang = self.embed_ang(x_ang)
        return h_atm, h_bnd, h_ang


class Processor(nn.Module):
    """ALIGNN Processor.
    """
    def __init__(self, num_convs, dim):
        super().__init__()
        self.num_convs = num_convs
        self.dim = dim

        self.atm_bnd_convs = nn.ModuleList([copy.deepcopy(MeshGraphNetsConv(dim, dim)) for _ in range(num_convs)])
        self.bnd_ang_convs = nn.ModuleList([copy.deepcopy(MeshGraphNetsConv(dim, dim)) for _ in range(num_convs)])

    def forward(self, h_atm:Tensor, h_bnd:Tensor, h_ang:Tensor, edge_index_bnd:Tensor, edge_index_ang:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        for bnd_ang_conv, atm_bnd_conv in zip(self.bnd_ang_convs, self.atm_bnd_convs):
            h_bnd, h_ang = bnd_ang_conv(h_bnd, edge_index_ang, h_ang)
            h_atm, h_bnd = atm_bnd_conv(h_atm, edge_index_bnd, h_bnd)
        return h_atm, h_bnd, h_ang


class Decoder(nn.Module):
    def __init__(self, node_dim, out_dim):
        super().__init__()
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.decoder = MLP([node_dim, node_dim, out_dim], act=nn.SiLU())

    def forward(self, h_atm:Tensor) -> Tensor:
        return self.decoder(h_atm)


class ALIGNN(nn.Module):
    def __init__(self, encoder, processor, decoder):
        super().__init__()
        self.encoder   = encoder
        self.processor = processor
        self.decoder   = decoder
    
    def forward(self, x_atm:Tensor, x_bnd:Tensor, x_ang:Tensor, edge_index_bnd:Tensor, edge_index_ang:Tensor) -> Tensor:
        h_atm, h_bnd, h_ang = self.encoder(x_atm, x_bnd, x_ang)
        h_atm, h_bnd, h_ang = self.processor(h_atm, h_bnd, h_ang, edge_index_bnd, edge_index_ang)
        return self.decoder(h_atm)
