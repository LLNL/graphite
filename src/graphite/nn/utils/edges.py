import torch


def add_edges(edge_index, new_edge_index, num_nodes):
    # Adjacency matrix of the original graph
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=edge_index.device)
    A[edge_index.split(1)] = True

    # Add the new edges to adjacency matrix
    A[new_edge_index.split(1)] = True

    # Convert from adjacency matrix back into edge indices in COO format
    return A.nonzero().T


def add_edges_v2(edge_index, new_edge_index):
    A, B = edge_index, new_edge_index
    # Mask array for removing cols in B that are duplicates to cols in A
    mask = (A[:, None, :] - B[..., None]).abs().sum(0).gt(0).all(1)
    return torch.hstack([A, B[:, mask]])


def edge_set(edge_index):
    return set(map(tuple, edge_index.T.numpy()))


def mask_edges(edge_index, edge_index_to_mask, num_nodes):
    assert edge_set(edge_index) >= edge_set(edge_index_to_mask), "'edge_index' must be a superset of 'edge_index_to_mask'"

    # Mask matrix in the same shape as adjacency matrix. Describes which edges are masked as 'True'
    M = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=edge_index.device)
    M[edge_index_to_mask.split(1)] = True
    
    # Get mask array of shape (data.edge_index.shape[1],) with True values for the edge subset
    return M[edge_index.split(1)].flatten()