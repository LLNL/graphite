def downselect_edges(edge_index, pos, cutoff, box=0.0):
    """Given a particle data with pre-computed edges/bonds,
    downselect the edges that are shorter than the specified cutoff.
    
    Args:
        edge_index: Pre-computed graph edges (in COO format).
        pos: Particle positions.
        cutoff: Cutoff value.
        box: An array of positive numbers describing the box dimensions
        if the particles lie in a periodic box. Otherwise leave it as zero.
    """
    i, j = edge_index
    edge_vec = pos[j] - pos[i]
    edge_vec -= (edge_vec >  box/2) * box
    edge_vec += (edge_vec < -box/2) * box
    mask = edge_vec.norm(dim=1) < cutoff
    return edge_index[:, mask], edge_vec[mask]
