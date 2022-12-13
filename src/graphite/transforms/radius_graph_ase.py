import torch
import numpy as np

from torch_geometric.transforms import BaseTransform

from ase.neighborlist import primitive_neighbor_list


class RadiusGraph_ase(BaseTransform):
    """Similar to PyG's RadiusGraph, but implemented with ASE's cell list algorithm.

    Only works for a single data object, not batches of data.

    Args:
        cutoff (float): Cutoff distance within which pairs of nodes would
            be considered connected.
    """
    def __init__(self, cutoff):
        self.cutoff = cutoff
    
    def __call__(self, data):
        pos, cell, pbc, numbers = data.pos, data.cell, data.pbc, data.numbers

        i, j, D = primitive_neighbor_list('ijD', pbc=pbc, cell=cell, positions=pos.cpu().numpy(), cutoff=self.cutoff, numbers=numbers)
        edge_index = np.stack((i, j))

        data.edge_index = torch.tensor(edge_index, dtype=torch.long)
        data.edge_attr  = torch.tensor(D,          dtype=torch.float)
        return data
    
    def __repr__(self):
        return f'{self.__class__.__name__}(cutoff={self.cutoff})'
