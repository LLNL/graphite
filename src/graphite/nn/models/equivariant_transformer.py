import torch
from torch import nn
from torch_geometric.utils import scatter
import copy

from ..mlp import MLP
from ..convs import EquivariantTransformerLayer
from ..basis import GaussianRandomFourierFeatures

# Typing
from torch import Tensor
from typing import List, Optional, Tuple


class Encoder(nn.Module):
    def __init__(self, num_species: int, node_dim: int, init_edge_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.num_species   = num_species
        self.node_dim      = node_dim
        self.init_edge_dim = init_edge_dim
        self.edge_dim      = edge_dim
        
        self.embed_atom = nn.Embedding(num_species, node_dim)
        self.embed_bond = MLP([init_edge_dim, edge_dim, edge_dim], act=nn.SiLU())
        self.phi_s = MLP([node_dim*2 + edge_dim, node_dim, node_dim], act=nn.SiLU())
        self.phi_h = MLP([node_dim*2,            node_dim, node_dim], act=nn.SiLU())
        self.phi_v = MLP([node_dim*2 + edge_dim, node_dim, node_dim], act=nn.SiLU())

    def forward(self, species: Tensor, pos: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        num_nodes = len(species)
        i = edge_index[0]
        j = edge_index[1]
        edge_vec = pos[j] - pos[i]
        edge_len = torch.linalg.norm(edge_vec, dim=1, keepdim=True)
        
        # Embed node and edge features
        f         = self.embed_atom(species)
        edge_attr = self.embed_bond(edge_len)

        # Convolve node features
        e  = torch.cat([f[i], f[j], edge_attr], dim=-1)
        h0 = self.phi_h(torch.cat([
            f, scatter(self.phi_s(e) * f[i], index=j, dim=0, dim_size=num_nodes)
        ], dim=-1))

        # Initialize vector features
        v0 = scatter(edge_vec[:, None, :] * self.phi_v(e)[:, :, None], index=j, dim=0, dim_size=num_nodes)
        return h0, v0, edge_attr


class Processor(nn.Module):
    def __init__(self, num_convs: int, node_dim: int, num_heads: int, ff_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.num_convs = num_convs
        self.node_dim  = node_dim
        self.ff_dim    = ff_dim
        self.edge_dim  = edge_dim
        self.convs = nn.ModuleList(
            [copy.deepcopy(EquivariantTransformerLayer(node_dim, num_heads, ff_dim, edge_dim)) for _ in range(num_convs)]
        )

    def forward(self, h: Tensor, v: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_len: Tensor) -> Tuple[Tensor, Tensor]:
        for conv in self.convs:
            h, v = conv(h, v, edge_index, edge_attr, edge_len)
        return h, v


class Decoder(nn.Module):
    def __init__(self, dim: int, num_scalar_out: int, num_vector_out: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_scalar_out = num_scalar_out
        self.num_vector_out = num_vector_out
        self.Oh = nn.Parameter(torch.randn(dim, num_scalar_out))
        self.Ov = nn.Parameter(torch.randn(dim, num_vector_out))

    def forward(self, h:Tensor, v: Tensor) -> Tensor:
        h_out = h @ self.Oh
        v_out = torch.einsum('ndi, df -> nfi', v, self.Ov)
        return h_out, v_out.squeeze()

    def extra_repr(self) -> str:
        return f'(Oh): tensor({list(self.Oh.shape)}, requires_grad={self.Oh.requires_grad}) \n' \
             + f'(Ov): tensor({list(self.Ov.shape)}, requires_grad={self.Ov.requires_grad})'


class EquivariantTransformer(nn.Module):
    def __init__(self, encoder, processor, decoder):
        super().__init__()
        self.encoder   = encoder
        self.processor = processor
        self.decoder   = decoder
    
    def forward(self, species: Tensor, pos: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        h, v, edge_attr = self.encoder(species, pos, edge_index)
        h, v = self.processor(h, v, edge_index, edge_len, edge_attr)
        return self.decoder(h, v)
