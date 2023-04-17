import torch
from torch_geometric.transforms import BaseTransform


class MaskEdges(BaseTransform):
    """Given a graph data with a set of edges `data.edge_index`, label a subset of the edges
    according to `edge_index_subset` with a binary mask `data.edge_subset_mask`.
    
    `data.edge_index` must be a superset of `edge_index_subset`.
    
    Args:
        edge_index_subset (2xM tensor): Edge indices in COO format.
        num_nodes (int): Number of nodes in the graph.
    """
    def __init__(self, edge_index_subset, num_nodes):
        self.edge_index_subset = edge_index_subset
        self.num_nodes = num_nodes

    def __call__(self, data):
        num_nodes = self.num_nodes

        # Mask matrix in the same shape as adjacency matrix. Describes which edges are masked as 'True'
        M = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=data.edge_index.device)
        M[self.edge_index_subset.split(1)] = True
        
        # Get mask array of shape (data.edge_index.shape[1],) with True values for the edge subset
        data.edge_subset_mask = M[data.edge_index.split(1)].flatten()

        return data