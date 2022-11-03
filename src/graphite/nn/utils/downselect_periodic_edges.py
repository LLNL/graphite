def downselect_periodic_edges(edge_index, pos, cutoff, box):
    """Given a periodic box of particles with pre-computed edges/bonds,
    downselect the edges that are shorter than the specified cutoff.
    
    Args:
        edge_index: Pre-computed graph edges (in COO format).
        pos: Particle positions.
        cutoff: Cutoff value.
        box: The size of the periodic box. Must be an array of positive numbers.
    """
    i, j = edge_index
    edge_vec = pos[j] - pos[i]
    
    for i in range(len(box)):
        edge_vec[:,i] -= (edge_vec[:,i] >  box[i]/2) * box[i]
        edge_vec[:,i] += (edge_vec[:,i] < -box[i]/2) * box[i]
    
    mask = edge_vec.norm(dim=1) < cutoff    
    return edge_index[:, mask], edge_vec[mask]
