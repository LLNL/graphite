import torch
import math
from torch import nn
from torch_geometric.utils import scatter
from ..utils import graph_softmax

# Typing
from torch import Tensor
from typing import List, Optional, Tuple

from ..mlp import MLP


class NodeAttention(nn.Module):
    def __init__(self, dim, num_heads, edge_dim):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.lin_Q = nn.Linear(dim, dim)
        self.lin_K = nn.Linear(dim, dim)
        self.lin_V = nn.Linear(dim, dim)
        self.lin_O = nn.Linear(dim, dim)
        self.edge_bias = MLP([edge_dim, edge_dim, num_heads], act=nn.SiLU())
    
    def forward(self, x:Tensor, edge_index:Tensor, edge_attr:Tensor) -> Tensor:
        i = edge_index[0]
        j = edge_index[1]

        # Query, Key, and Value embedding
        query = self.lin_Q(x)[j]
        key   = self.lin_K(x)[i]
        value = self.lin_V(x)[i]

        # Multi-head attention
        d_k = self.dim // self.num_heads
        query, key, value = [x.view(-1, self.num_heads, d_k) for x in (query, key, value)]
        scores = (query * key).sum(dim=-1) / math.sqrt(d_k) + self.edge_bias(edge_attr)
        alpha = graph_softmax(scores, index=j, num_nodes=x.size(0))
        attn_out = (alpha[..., None] * value).view(-1, self.dim)
        attn_out = scatter(attn_out, index=j, dim=0, dim_size=x.size(0))
        return self.lin_O(attn_out)


class GraphormerConv(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, edge_dim):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ff_dim    = ff_dim
        self.edge_dim  = edge_dim
        self.node_attn = NodeAttention(dim, num_heads, edge_dim)
        self.norm1     = nn.LayerNorm(dim)
        self.norm2     = nn.LayerNorm(dim)
        self.ffn       = MLP([dim, ff_dim, dim], act=nn.SiLU())
    
    def forward(self, x:Tensor, edge_index:Tensor, edge_attr:Tensor) -> Tensor:
        out = self.norm1(x   + self.node_attn(x, edge_index, edge_attr))
        out = self.norm2(out + self.ffn(out))
        return out
    
    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim}, num_heads={self.num_heads}, ff_dim={self.ff_dim}, edge_dim={self.edge_dim})'


class EquivariantVectorPrediction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin_Q = nn.Linear(dim, dim)
        self.lin_K = nn.Linear(dim, dim)
        self.lin_V = nn.Linear(dim, dim)
        self.edge_bias = MLP([1+3, dim, dim, 1], act=nn.SiLU())
        self.w_F = nn.Linear(dim, 1)
    
    def forward(self, x:Tensor, edge_index:Tensor, edge_vec:Tensor) -> Tensor:
        i = edge_index[0]
        j = edge_index[1]

        # Query, Key, and Value embeddings
        query = self.lin_Q(x)[j]
        key   = self.lin_K(x)[i]
        value = self.lin_V(x)[i]

        # Edge features for edge biases to the attention scores
        edge_len = torch.linalg.norm(edge_vec, dim=-1, keepdim=True)
        edge_attr = torch.hstack([edge_vec, edge_len])

        # Single-head attention
        scores = (query * key).sum(dim=-1) / math.sqrt(self.dim) + self.edge_bias(edge_attr).squeeze()
        alpha = softmax(scores, index=j, dim=0, num_nodes=x.size(0))
        vec_out = scatter(
            alpha[..., None] * self.w_F(value) * edge_vec,
            index=j, dim=0, dim_size=x.size(0),
        )
        return vec_out

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim})'