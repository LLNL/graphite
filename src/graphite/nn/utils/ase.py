import torch
import numpy as np

# Typing
from numpy import ndarray
from typing import Optional, Tuple, List

from ase.neighborlist import primitive_neighbor_list


def ase_radius_graph(x: Tensor, r: float, numbers: Optional[ndarray] = None, cell: ndarray = np.diag([1.,1.,1.]), pbc: List[bool] = [False]*3) -> Tuple[Tensor, Tensor]:
    """Computes graph edges based on a cutoff radius for 3D structure data with periodic boundaries.
    This implementation uses ASE's neighbor list algorithm, which accounts for periodic boundaries.
    
    Args:
        x (Tensor): Point cloud coordinates with shape (N, 3) where N is the number of nodes/particles.
        r (float): Cutoff radius.
        numbers (ndarray, optional): 1D vector of atomic numbers.
        cell (ndarray, optional): Periodic cell dimensions with shape (3, 3).
        pbc (List[bool], optional): 1D vector indicating which cell boundary is periodic.

    Returns:
        edge_index (Tensor): Edge indices with shape (2, E) where E is the number of (directed) edges.
        edge_vec (Tensor): Edge vectors with shape (E, 3).

    Notes:
        - Does not work for batched inputs.
    """
    x_  = x.clone().detach().cpu().numpy()
    i, j, S = primitive_neighbor_list('ijS', positions=x_, cell=cell, cutoff=r, pbc=pbc, numbers=numbers)
    i    = torch.tensor(i,    dtype=torch.long,  device=x.device)
    j    = torch.tensor(j,    dtype=torch.long,  device=x.device)
    S    = torch.tensor(S,    dtype=torch.float, device=x.device)
    cell = torch.tensor(cell, dtype=torch.float, device=x.device)
    edge_index = torch.stack([i, j])
    edge_vec = x[j] - x[i] + S@cell
    return edge_index, edge_vec