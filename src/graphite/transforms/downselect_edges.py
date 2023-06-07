import torch
from torch_geometric.transforms import BaseTransform


class DownselectEdges(BaseTransform):
    """Given a particle data with a set of edges/bonds,
    downselect the edges that are shorter than the specified cutoff.
    """
    def __init__(self, cutoff, cell=None):
        self.cutoff = cutoff
    
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        mask = (edge_attr.norm(dim=1) <= self.cutoff)
        data.edge_index = edge_index[:, mask]
        data.edge_attr  = edge_attr[mask]
        return data
    
    def __repr__(self):
        return f'{self.__class__.__name__}(cutoff={self.cutoff})'