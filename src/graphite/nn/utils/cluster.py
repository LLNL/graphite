import torch
from torch import tensor as t

# Typing
from torch import Tensor
from typing import Optional, Tuple, List


@torch.jit.script
def _knn_graph(x: Tensor, k: int, loop: bool = False) -> Tensor:
    dist = torch.cdist(x, x)
    dist += torch.eye(x.size(0), device=x.device) * (1. - float(loop)) * 1e6
    j = torch.repeat_interleave(torch.arange(x.size(0), device=x.device), k)
    _, i = dist.topk(k, largest=False)
    return torch.stack([i.view(-1), j])


def knn_graph(x: Tensor, k: int, batch: Optional[Tensor] = None, loop: bool = False) -> Tensor:
    """Computes graph edges based on k nearest neighbors for point cloud data.
    
    Args:
        x (Tensor): Point cloud coordinates with shape (N, D) where N is the number of nodes/particles,
            and D is the dimensionality of the coordinate space.
        k (float): The number of nearest neighbors for each node.
        batch (Tensor, optional): Batch vector for batched inputs. See PyG documentation for further explanation.
        loop (bool): Whether to include self-loops.

    Returns:
        edge_index (Tensor): Edge indices with shape (2, E) where E is the number of (directed) edges.
    """
    if batch is None:
        return _knn_graph(x, k, loop)

    bsize = int(batch.max()) + 1
    arange = torch.arange(bsize + 1, device=x.device)
    ptr = torch.bucketize(arange, batch)
    
    edge_index = []
    for b, inc in zip(range(bsize), ptr[:-1]):
        edge_index_ = _knn_graph(x[batch == b], k, loop)
        edge_index.append(edge_index_ + inc)

    return torch.hstack(edge_index)


@torch.jit.script
def _radius_graph(x: Tensor, r: float, loop: bool = False) -> Tensor:
    dist = torch.cdist(x, x)
    dist += torch.eye(x.size(0), device=x.device) * (1. - float(loop)) * (r + 1.)
    edge_index = torch.nonzero(dist < r).T
    return edge_index


def radius_graph(x: Tensor, r: float, batch: Optional[Tensor] = None, loop: bool = False) -> Tensor:
    """Computes graph edges based on a cutoff radius for point cloud data.
    
    Args:
        x (Tensor): Point cloud coordinates with shape (N, D) where N is the number of nodes/particles,
            and D is the dimensionality of the coordinate space.
        r (float): Cutoff radius.
        batch (Tensor, optional): Batch vector for batched inputs. See PyG documentation for further explanation.
        loop (bool): Whether to include self-loops.

    Returns:
        edge_index (Tensor): Edge indices with shape (2, E) where E is the number of (directed) edges.
    """
    if batch is None:
        return _radius_graph(x, r, loop)

    bsize = int(batch.max()) + 1
    arange = torch.arange(bsize + 1, device=x.device)
    ptr = torch.bucketize(arange, batch)
    
    edge_index = []
    for b, inc in zip(range(bsize), ptr[:-1]):
        edge_index_ = _radius_graph(x[batch == b], r, loop)
        edge_index.append(edge_index_ + inc)

    return torch.hstack(edge_index)


def periodic_radius_graph(x: Tensor, r: float, cell: Tensor, loop: bool = False) -> Tuple[Tensor, Tensor]:
    """Computes graph edges based on a cutoff radius for point cloud data with periodic boundaries.
    This implementation is bruteforce with O(N^2) complexity (per batch), but is very quick for small scale data.
    
    Args:
        x (Tensor): Point cloud coordinates with shape (N, D) where N is the number of nodes/particles,
            and D is the dimensionality of the coordinate space.
        r (float): Cutoff radius.
        cell (Tensor): Periodic cell dimensions with shape (D, D). Normally for 3D data the shape is (3, 3).
        loop (bool): Whether to include self-loops.

    Returns:
        edge_index (Tensor): Edge indices with shape (2, E) where E is the number of (directed) edges.
        edge_vec (Tensor): Edge vectors with shape (E, D).

    Notes:
        - Does not work for batched inputs.
        - Not tested with D != 3 dimensionality.
        - Not accurate for cells that are very oblique.
    """
    inv_cell = torch.linalg.pinv(cell)
    
    vec = x[None,:,:] - x[:,None,:]
    vec = vec - torch.round(vec @ inv_cell) @ cell
    dist = torch.linalg.norm(vec, dim=-1)
    dist += torch.eye(x.size(0), device=x.device) * (1. - float(loop)) * (r + 1.)
    edge_index = torch.nonzero(dist < r).T
    i, j = edge_index
    edge_vec = vec[i, j]
    return edge_index, edge_vec


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


def periodic_radius_graph_v2(x: Tensor, r: float, box: Tensor) -> Tuple[Tensor, Tensor]:
    """Computes graph edges based on a cutoff radius for point cloud data with periodic boundaries.
    This version is implemented with the cell list algorithm and should have linear complexity.

    Args:
        x (Tensor): Point cloud coordinates.
        r (float): Cutoff radius.
        box (Tensor): The size of the periodic box. Must be an array of positive numbers.

    Notes:
        - Does not work for batched inputs.
        - Works for 2D and 3D.
        - Assumes the periodic box is an orthorombic box, not an arbitrary triclinic cell.
        - The paritcle positions can be outside the box. They will get wrapped back inside.

    References:
        - https://en.wikipedia.org/wiki/Cell_lists
        - https://aiichironakano.github.io/cs596/01-1LinkedListCell.pdf
        - https://wiki.fysik.dtu.dk/ase/_modules/ase/neighborlist.html
    """
    num_atoms, num_dims = x.shape

    # Determine number of cells in each dimension and the cell dimensions.
    shape     = box.div(r).to(torch.long)
    cell_dims = box / shape
    num_cells = shape.prod()
    
    # Determine the atoms' cell coordinates and their shifts if outside the simulation box.
    coords    = torch.div(x, cell_dims, rounding_mode='floor').to(torch.long)
    shifts    = cell_shift(coords, shape)

    # If any atom is outside the box, wrap the positions and the cell coordinates back into the box.
    x      = x - box*shifts
    coords = wrap_cell_coord(coords, shape)

    # Used for Cartesian-to-linear index conversion.
    cumprod   = torch.cat((t([1], device=shape.device), shape[:-1].cumprod(dim=0)))

    # Convert the atoms' cell coordinates to linear indices
    indices   = (cumprod*coords).sum(dim=-1)

    # Construct the `cells` matrix, where each row is a cell, and the columns
    # denote the atom indices that occupy the cell. Indices of -1 are used for padding.
    max_atoms_per_cell = torch.bincount(indices).max()
    cells = -torch.ones(num_cells, max_atoms_per_cell, dtype=torch.long, device=x.device)
    counts = torch.zeros(num_cells, dtype=torch.long)
    for atom_idx, cell_idx in enumerate(indices):
        n = counts[cell_idx]
        cells[cell_idx, n] = atom_idx
        counts[cell_idx] += 1

    # Determine the center cell indices and for each the correpsonding neighbor cell indices
    center_coords  = torch.unique(coords, dim=0)
    center_indices = (cumprod*center_coords).sum(dim=-1)
    nbr_disp       = torch.cartesian_prod(*torch.arange(-1, 2).expand(num_dims, 3)).to(x.device)
    nbr_coords     = center_coords.unsqueeze(1) + nbr_disp.unsqueeze(0)
    nbr_shifts     = cell_shift(nbr_coords, shape)
    nbr_coords     = wrap_cell_coord(nbr_coords, shape)
    nbr_indices    = (cumprod*nbr_coords).sum(dim=-1)

    # Used for the Cartesian product between atom indices in center cell vs. neighbor cell.
    ii, jj = torch.cartesian_prod(
        torch.arange(max_atoms_per_cell),
        torch.arange(max_atoms_per_cell)
    ).T.to(x.device)

    # Loop over cells with atoms in it, compute vectors between atoms in the cell and
    # the atoms in the neighbor cells, accounting for periodic boundaries.
    src, dst, vec = [], [], []
    for c1, c2, s in zip(center_indices, nbr_indices, nbr_shifts):
        # i, j, v = connect_neighbors(cells, c1, c2, s, x, box, ii, jj)
        i = cells[c1].unsqueeze(0).expand(len(nbr_disp), -1)
        j = cells[c2]
        i = torch.take_along_dim(i, ii[None, :], dim=-1)
        j = torch.take_along_dim(j, jj[None, :], dim=-1)
        v = (x[j] + s.unsqueeze(1)*box) - x[i]

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
    mask = vec.norm(dim=-1) < r
    src, dst, vec = src[mask], dst[mask], vec[mask]

    # Remove self-connections
    mask = (src != dst)
    src, dst, vec = src[mask], dst[mask], vec[mask]

    return torch.stack((src, dst)), vec
