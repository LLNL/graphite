import torch
from torch_geometric.transforms import BaseTransform


class LabelStrongBonds(BaseTransform):
    """Given a particle data with a set of global bonds/edges `data.edge_index`,
    label those that are strong chemical bonds according to `strong_edge_index`
    with a mask `mask_strong_bnd`.
    
    `data.edge_index` must be a superset of `strong_edge_index`.
    
    Args:
        strong_edge_index: Edge indices of strong chemical bonds.
    """
    def __init__(self, strong_edge_index):
        self.strong_edge_index = strong_edge_index

    def __call__(self, data):
        num_nodes = data.x.size(0)

        # Flag matrix similar to an adjacency matrix, but describes which edges are strong
        M = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=data.edge_index.device)
        M[self.strong_edge_index.split(1)] = True
        
        # Get mask array of shape (all_edge_index.shape[1],) with True values for strong bonds
        data.mask_strong_bnd = M[data.edge_index.split(1)].flatten()

        return data