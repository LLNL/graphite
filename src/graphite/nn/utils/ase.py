import torch
import numpy as np

from ase.neighborlist import primitive_neighbor_list


def ase_radius_graph(pos, cutoff, numbers=None, cell=np.diag([1.,1.,1.]), pbc=[False]*3):
    """Computes graph edges based on a cutoff radius for 3D structure data with periodic boundaries.
    Returns the edge indices `edge_index` and the edge vectors `edge_vec`.

    This implementation uses ASE's neighbor list algorithm, which accounts for periodic boundaries.
    """
    pos_  = pos.clone().detach().cpu().numpy()
    i, j, S = primitive_neighbor_list('ijS', positions=pos_, cell=cell, cutoff=cutoff, pbc=pbc, numbers=numbers)
    i = torch.tensor(i, dtype=torch.long,  device=pos.device)
    j = torch.tensor(j, dtype=torch.long,  device=pos.device)
    S = torch.tensor(S, dtype=torch.float, device=pos.device)
    edge_index = torch.stack([i, j])
    edge_vec = pos[j] - pos[i] + S@cell
    return edge_index, edge_vec