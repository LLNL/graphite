import torch
from torch import tensor as t


# def coord2index(coord, shape):
#     cumprod = torch.cat([t([1]), shape[:-1].cumprod(dim=0)])
#     return cumprod.dot(coord)


# def coords2indices(coords, shape):
#     cumprod = torch.cat([t([1]), shape[:-1].cumprod(dim=0)])
#     return (cumprod*coords).sum(dim=-1)


# def connect_neighbors(cells, c1, c2, s, pos, box, ii, jj):
#     # Look up atoms in center cell `c1` and those in neighbor cells `c2` (plural),
#     # connect all atom pairs and compute the edge vectors.
#     i = cells[c1].unsqueeze(0).expand(len(c2), -1)
#     j = cells[c2]
#     i = torch.take_along_dim(i, ii[None, :], dim=-1)
#     j = torch.take_along_dim(j, jj[None, :], dim=-1)
#     v = (pos[j] + s.unsqueeze(1)*box) - pos[i]

#     i, j, v = i.reshape(-1), j.reshape(-1), v.reshape(-1, pos.size(1))

#     # Remove superfluous pairs with atom index -1
#     mask = torch.logical_and(i != -1, j != -1)
#     return i[mask], j[mask], v[mask]


# def neighbor_cell_coords(coord):
#     return torch.cartesian_prod(*(torch.arange(n-1, n+2) for n in coord))


def wrap_cell_coord(coord, shape):
    return (coord + shape) % shape


def cell_shift(coord, shape):
    return torch.div(coord, shape, rounding_mode='floor')  # coord // shape


def unravel_index(idx, shape):
    coord = []
    for dim in reversed(shape):
        coord.append(idx % dim)
        idx = torch.div(idx, dim, rounding_mode='floor')
    return torch.stack(coord, dim=-1)


def periodic_radius_graph(pos, box, cutoff):
    """Computes graph edges based on a cutoff radius for point cloud data with periodic boundaries.
    Returns the edge indices `edge_index` and the edge vectors `edge_vec`.

    Args:
        pos: Particle positions.
        box: The size of the periodic box. Must be an array of positive numbers.
        cutoff: The cutoff distance/radius.

    Notes:
    - Works for 2D and 3D.
    - The paritcle positions can be outside the box. They will get wrapped back inside.

    References:
    - https://en.wikipedia.org/wiki/Cell_lists
    - https://aiichironakano.github.io/cs596/01-1LinkedListCell.pdf
    - https://wiki.fysik.dtu.dk/ase/_modules/ase/neighborlist.html
    """
    num_atoms, num_dims = pos.shape

    # Determine number of cells in each dimension and the cell dimensions.
    shape     = box.div(cutoff).to(torch.long)
    cell_dims = box / shape
    num_cells = shape.prod()
    
    # Determine the atoms' cell coordinates and their shifts if outside the simulation box.
    coords    = torch.div(pos, cell_dims, rounding_mode='floor').to(torch.long)
    shifts    = cell_shift(coords, shape)

    # If any atom is outside the box, wrap the positions and the cell coordinates back into the box.
    pos       = pos - box*shifts
    coords    = wrap_cell_coord(coords, shape)

    # Used for Cartesian-to-linear index conversion.
    cumprod   = torch.cat((t([1], device=shape.device), shape[:-1].cumprod(dim=0)))

    # Convert the atoms' cell coordinates to linear indices
    indices   = (cumprod*coords).sum(dim=-1)

    # Construct the `cells` matrix, where each row is a cell, and the columns
    # denote the atom indices that occupy the cell. Indices of -1 are used for padding.
    max_atoms_per_cell = torch.bincount(indices).max()
    cells = -torch.ones(num_cells, max_atoms_per_cell, dtype=torch.long, device=pos.device)
    counts = torch.zeros(num_cells, dtype=torch.long)
    for atom_idx, cell_idx in enumerate(indices):
        n = counts[cell_idx]
        cells[cell_idx, n] = atom_idx
        counts[cell_idx] += 1

    # Determine the center cell indices and for each the correpsonding neighbor cell indices
    center_coords  = torch.unique(coords, dim=0)
    center_indices = (cumprod*center_coords).sum(dim=-1)
    nbr_disp       = torch.cartesian_prod(*torch.arange(-1, 2).expand(num_dims, 3)).to(pos.device)
    nbr_coords     = center_coords.unsqueeze(1) + nbr_disp.unsqueeze(0)
    nbr_shifts     = cell_shift(nbr_coords, shape)
    nbr_coords     = wrap_cell_coord(nbr_coords, shape)
    nbr_indices    = (cumprod*nbr_coords).sum(dim=-1)

    # Used for the Cartesian product between atom indices in center cell vs. neighbor cell.
    ii, jj = torch.cartesian_prod(
        torch.arange(max_atoms_per_cell),
        torch.arange(max_atoms_per_cell)
    ).T.to(pos.device)

    # Loop over cells with atoms in it, compute vectors between atoms in the cell and
    # the atoms in the neighbor cells, accounting for periodic boundaries.
    src, dst, vec = [], [], []
    for c1, c2, s in zip(center_indices, nbr_indices, nbr_shifts):
        # i, j, v = connect_neighbors(cells, c1, c2, s, pos, box, ii, jj)
        i = cells[c1].unsqueeze(0).expand(len(nbr_disp), -1)
        j = cells[c2]
        i = torch.take_along_dim(i, ii[None, :], dim=-1)
        j = torch.take_along_dim(j, jj[None, :], dim=-1)
        v = (pos[j] + s.unsqueeze(1)*box) - pos[i]

        i, j, v = i.reshape(-1), j.reshape(-1), v.reshape(-1, num_dims)

        # Remove superfluous pairs with atom index -1
        mask = torch.logical_and(i != -1, j != -1)
        src.append(i[mask])
        dst.append(j[mask])
        vec.append(v[mask])
    
    src = torch.cat(src)
    dst = torch.cat(dst)
    vec = torch.cat(vec, dim=0)

    # Remove edges that are longer than the cutoff
    mask = vec.norm(dim=-1) < cutoff
    src, dst, vec = src[mask], dst[mask], vec[mask]

    # Remove self-connections
    mask = (src != dst)
    src, dst, vec = src[mask], dst[mask], vec[mask]

    return torch.stack((src, dst)), vec
