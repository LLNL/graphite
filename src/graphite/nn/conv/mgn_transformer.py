import torch
from torch import nn
import math
from torch_geometric.utils import scatter
from ..utils import graph_softmax

# Typing
from torch import Tensor
from typing import List, Optional, Tuple

from ..mlp import MLP


class EdgeProcessor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(MLP([dim*3, dim, dim, dim], act=nn.SiLU()), nn.LayerNorm(dim))

    def forward(self, x:Tensor, edge_index:Tensor, edge_attr:Tensor) -> Tensor:
        i = edge_index[0]
        j = edge_index[1]
        out  = self.edge_mlp(torch.cat([x[i], x[j], edge_attr], dim=-1))
        out += edge_attr
        return out


class EdgeToNodeAttention(nn.Module):
    def __init__(self, dim, num_heads, cutoff):
        super().__init__()
        self.dim       = dim
        self.num_heads = num_heads
        self.cutoff    = cutoff
        self.lin_Q     = nn.Linear(dim, dim)
        self.lin_K     = nn.Linear(dim, dim)
        self.lin_V     = nn.Linear(dim, dim)
        self.lin_O     = nn.Linear(dim, dim)
    
    def forward(self, x:Tensor, edge_index:Tensor, edge_attr:Tensor, edge_len:Tensor) -> Tensor:
        i = edge_index[0]
        j = edge_index[1]

        # Query, Key, and Value embedding
        query = self.lin_Q(x)[j]
        key   = self.lin_K(edge_attr)
        value = self.lin_V(edge_attr)
        
        # Smooth cutoff function
        phi_cutoff = 0.5 * ( torch.cos(torch.pi*edge_len/self.cutoff) + 1)

        # Multi-head attention
        d_k = self.dim // self.num_heads
        query, key, value = [x.view(-1, self.num_heads, d_k) for x in (query, key, value)]
        scores = (query * key).sum(dim=-1) / math.sqrt(d_k)
        alpha = graph_softmax(scores, index=j, num_nodes=x.size(0))
        attn_out = (alpha[..., None] * value * phi_cutoff[..., None]).view(-1, self.dim)
        attn_out = scatter(attn_out, index=j, dim=0, dim_size=x.size(0))
        return self.lin_O(attn_out)


class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, cutoff):
        super().__init__()
        self.dim       = dim
        self.num_heads = num_heads
        self.cutoff    = cutoff
        self.e2n_attn  = EdgeToNodeAttention(dim, num_heads, cutoff)
        self.norm1     = nn.LayerNorm(dim)
        self.norm2     = nn.LayerNorm(dim)
        self.ffn       = MLP([dim, ff_dim, dim], act=nn.SiLU())
        # self.node_mlp = nn.Sequential(MLP([dim*2, dim, dim, dim], act=nn.SiLU()), nn.LayerNorm(dim))
    
    def forward(self, x:Tensor, edge_index:Tensor, edge_attr:Tensor, edge_len:Tensor) -> Tensor:
        # out = self.e2n_attn(x, edge_index, edge_attr, edge_len)
        # out = torch.cat([x, out], dim=-1)
        # out = x + self.node_mlp(out)
        out = self.norm1(x   + self.e2n_attn(x, edge_index, edge_attr, edge_len))
        out = self.norm2(out + self.ffn(out))
        return out


class MGNTransformerConv(nn.Module):
    def __init__(self, dim, ff_dim, num_heads, cutoff):
        super().__init__()
        self.dim       = dim
        self.ff_dim    = ff_dim
        self.num_heads = num_heads
        self.cutoff    = cutoff
        self.edge_processor = EdgeProcessor(dim)
        self.node_processor = TransformerLayer(dim, num_heads, ff_dim, cutoff)
    
    def forward(self, x:Tensor, edge_index:Tensor, edge_attr:Tensor, edge_len:Tensor) -> Tuple[Tensor, Tensor]:
        edge_attr = self.edge_processor(x, edge_index, edge_attr)
        x         = self.node_processor(x, edge_index, edge_attr, edge_len)
        return x, edge_attr

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim}, ff_dim={self.ff_dim}, num_heads={self.num_heads}, cutoff={self.cutoff})'
