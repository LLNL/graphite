from torch_geometric.transforms import BaseTransform

from ..nn import downselect_periodic_edges


class DownselectPeriodicEdges(BaseTransform):
    def __init__(self, cutoff):
        self.cutoff = cutoff
    
    def __call__(self, data):
        edge_index, pos, box = data.edge_index, data.pos, data.box

        edge_index, edge_vec = downselect_periodic_edges(edge_index, pos, self.cutoff, box)
        
        data.edge_index = edge_index
        data.edge_attr  = edge_vec
        return data
    
    def __repr__(self):
        return f'{self.__class__.__name__}(cutoff={self.cutoff})'