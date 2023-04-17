import torch
from torch_geometric.transforms import BaseTransform


class AddEdges(BaseTransform):
    """Given a graph data with a set of edges `data.edge_index`, add another set of edges
    into the graph while discarding duplicates.
        
    Args:
        edge_index (2xM tensor): Edge indices (in COO format) to be added.
        num_nodes (int): Number of nodes in the graph.
    """
    def __init__(self, edge_index, num_nodes):
        self.edge_index = edge_index
        self.num_nodes = num_nodes

    def __call__(self, data):
        num_nodes = self.num_nodes

        # Adjacency matrix of original graph
        A = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=data.edge_index.device)
        A[data.edge_index.split(1)] = True

        # Add another set of edges
        A[self.edge_index.split(1)] = True

        # Convert from adjacency matrix back into edge indices in COO format
        data.edge_index = A.nonzero().T
        
        return data