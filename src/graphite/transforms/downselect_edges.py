from torch_geometric.transforms import BaseTransform

from ..nn import downselect_edges


class DownselectEdges(BaseTransform):
    """Given a particle data with a set of edges/bonds,
    downselect the edges that are shorter than the specified cutoff.

    Args:
        cutoff (float): Cutoff distance within which pairs of nodes would
            be considered connected.
    """
    def __init__(self, cutoff):
        self.cutoff = cutoff
    
    def __call__(self, data):
        edge_index, pos = data.edge_index, data.pos
        box = data.box if hasattr(data, 'box') else 0.0
        edge_index, edge_vec = downselect_edges(edge_index, pos, self.cutoff, box)
        data.edge_index = edge_index
        data.edge_attr  = edge_vec
        return data
    
    def __repr__(self):
        return f'{self.__class__.__name__}(cutoff={self.cutoff})'