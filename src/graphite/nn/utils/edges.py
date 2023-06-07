import torch


def add_edges(data, edge_index_to_add):
    num_nodes = data.pos.size(0)

    # Adjacency matrix of original graph
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=data.edge_index.device)
    A[data.edge_index.split(1)] = True

    # Add another set of edges
    A[edge_index_to_add.split(1)] = True

    # Convert from adjacency matrix back into edge indices in COO format
    data.edge_index = A.nonzero().T

    # Update edge attributes
    i, j = data.edge_index
    data.edge_attr = data.pos[j] - data.pos[i]
    
    return data


def mask_edges(data, edge_index_to_mask):
    num_nodes = data.pos.size(0)
    
    # Mask matrix in the same shape as adjacency matrix. Describes which edges are masked as 'True'
    M = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=data.edge_index.device)
    M[edge_index_to_mask.split(1)] = True
    
    # Get mask array of shape (data.edge_index.shape[1],) with True values for the edge subset
    data.edge_mask = M[data.edge_index.split(1)].flatten()

    return data