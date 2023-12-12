import torch

# Typing
from torch import Tensor
from typing import List, Optional, Tuple


def graph_scatter(src:Tensor, index:Tensor, num_nodes:int, reduce:str='sum') -> Tensor:
    index_ = index.unsqueeze(1).expand(-1, src.size(1))
    return torch.zeros(num_nodes, src.size(1), device=src.device).scatter_reduce(
        dim=0, index=index_, src=src, reduce=reduce)


def graph_softmax(src:Tensor, index:Tensor, num_nodes:int) -> Tensor:
    src_max = graph_scatter(src.detach(), index, num_nodes, reduce='amax')
    out = src - src_max[index]
    out = out.exp()
    out_sum = graph_scatter(out, index, num_nodes, reduce='sum') + 1e-16
    out_sum = out_sum[index]
    return out / out_sum