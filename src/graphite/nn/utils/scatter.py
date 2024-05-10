import torch
from torch_geometric.utils import scatter

# Typing
from torch import Tensor
from typing import List, Optional, Tuple


def graph_softmax(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    src_max = scatter(src, index=index, dim=0, dim_size=dim_size, reduce='max')
    out = src - src_max[index]
    out = out.exp()
    out_sum = scatter(out, index=index, dim=0, dim_size=dim_size, reduce='sum') + 1e-12
    out_sum = out_sum[index]
    return out / out_sum


def graph_scatter(src: Tensor, index: Tensor, dim_size: int, reduce: str = 'sum') -> Tensor:
    index_ = index.unsqueeze(1).expand(-1, src.size(1))
    return torch.zeros(dim_size, src.size(1), device=src.device).scatter_reduce(
        dim=0, index=index_, src=src, reduce=reduce)


def graph_softmax_v2(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    src_max = graph_scatter(src.detach(), index=index, dim_size=dim_size, reduce='amax')
    out = src - src_max[index]
    out = out.exp()
    out_sum = graph_scatter(out, index=index, dim_size=dim_size, reduce='sum') + 1e-12
    out_sum = out_sum[index]
    return out / out_sum