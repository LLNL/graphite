import torch
from torch import nn
from torch_geometric.utils import scatter
import math

from ..mlp import MLP
from ..utils import graph_softmax

# Typing
from torch import Tensor
from typing import List, Optional, Tuple


class EquivariantAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, edge_dim: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.dim       = dim
        self.num_heads = num_heads
        self.edge_dim  = edge_dim
        self.W_Q  = nn.Linear(dim, dim)
        self.W_K  = nn.Linear(dim, dim)
        self.W_Vh = nn.Linear(dim, dim)
        self.W_Vv = nn.Parameter(torch.empty(dim, dim))
        self.W_Oh = nn.Parameter(torch.empty(dim, dim))
        self.W_Ov = nn.Parameter(torch.empty(dim, dim))
        self.edge_bias = MLP([edge_dim, edge_dim, num_heads], act=nn.SiLU())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.W_Vv, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_Oh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_Ov, a=math.sqrt(5))

    def forward(self, h: Tensor, v: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_len: Tensor) -> Tuple[Tensor, Tensor]:
        num_nodes = h.size(0)
        i = edge_index[0]
        j = edge_index[1]

        # Query, key, value embedding
        d_k = self.dim // self.num_heads
        query   = self.W_Q(h).view( -1, self.num_heads, d_k)[j]
        key     = self.W_K(h).view( -1, self.num_heads, d_k)[i]
        value_h = self.W_Vh(h).view(-1, self.num_heads, d_k)[i]
        value_v = torch.einsum('ndi, df -> nfi', v, self.W_Vv).view(-1, self.num_heads, d_k, 3)[i]

        # Multi-head attention
        scores = (query * key).sum(dim=-1) / math.sqrt(d_k) - edge_len + self.edge_bias(edge_attr)
        alpha = graph_softmax(scores, index=j, dim_size=num_nodes)

        h = (alpha[..., None] * value_h).view(-1, self.dim)
        h = scatter(h, index=j, dim=0, dim_size=num_nodes)

        v = (alpha[..., None, None] * value_v).view(-1, self.dim, 3)
        v = scatter(v, index=j, dim=0, dim_size=num_nodes)

        dh = h @ self.W_Oh
        dv = torch.einsum('ndi, df -> nfi', v, self.W_Ov)
        return dh, dv

    def extra_repr(self) -> str:
        return f'(W_Vv): tensor({list(self.W_Vv.shape)}, requires_grad={self.W_Vv.requires_grad}) \n' \
             + f'(W_Oh): tensor({list(self.W_Oh.shape)}, requires_grad={self.W_Oh.requires_grad}) \n' \
             + f'(W_Ov): tensor({list(self.W_Ov.shape)}, requires_grad={self.W_Ov.requires_grad})'


class EquivariantFeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int) -> None:
        super().__init__()
        self.dim    = dim
        self.ff_dim = ff_dim
        self.W1   = nn.Parameter(torch.empty(dim, dim))
        self.W2   = nn.Parameter(torch.empty(dim, dim))
        self.ffn1 = MLP([dim*2, ff_dim, dim], act=nn.SiLU())
        self.ffn2 = MLP([dim*2, ff_dim, dim], act=nn.SiLU())
        self.norm = nn.LayerNorm(dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))

    def forward(self, h: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        v1 = torch.einsum('ndi, df -> nfi', v, self.W1)
        v2 = torch.einsum('ndi, df -> nfi', v, self.W2)
        v1_norm = torch.linalg.norm(v1, dim=-1)
        dh = self.ffn1(torch.cat([h, v1_norm], dim=-1))
        u  = self.ffn2(torch.cat([h, v1_norm], dim=-1))
        dv = self.norm(u)[..., None] * v2
        return dh, dv

    def extra_repr(self) -> str:
        return f'(W1): tensor({list(self.W1.shape)}, requires_grad={self.W1.requires_grad}) \n' \
             + f'(W2): tensor({list(self.W2.shape)}, requires_grad={self.W2.requires_grad}) '


class EquivariantTransformerLayer(nn.Module):
    """Equivariant transformer attention and feedforward layer.

    Reference: https://arxiv.org/pdf/2402.12714
    """
    def __init__(self, dim: int, num_heads: int, ff_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.attn = EquivariantAttention(dim, num_heads, edge_dim)
        self.feedforward = EquivariantFeedForward(dim, ff_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, h: Tensor, v: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_len: Tensor) -> Tuple[Tensor, Tensor]:
        # Attention
        dh, dv = self.attn(self.norm1(h), v, edge_index, edge_attr, edge_len)
        h = h + dh
        v = v + dv

        # Feedfoward
        dh, dv = self.feedforward(self.norm2(h), v)
        h = h + dh
        v = v + dv
        return h, v