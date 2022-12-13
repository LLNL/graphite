from torch_geometric.transforms import BaseTransform

from ..nn import periodic_radius_graph


class PeriodicRadiusGraph(BaseTransform):
    """Similar to PyG's RadiusGraph, but for periodic particle data.

    Only works for a single data object, not batches of data.

    Args:
        cutoff (float): Cutoff distance within which pairs of nodes would
            be considered connected.
    """
    def __init__(self, cutoff):
        self.cutoff = cutoff
    
    def __call__(self, data):
        pos, box = data.pos, data.box
        edge_index, edge_vec = periodic_radius_graph(pos, box, self.cutoff)
        data.edge_index = edge_index
        data.edge_attr  = edge_vec
        return data
    
    def __repr__(self):
        return f'{self.__class__.__name__}(cutoff={self.cutoff})'
