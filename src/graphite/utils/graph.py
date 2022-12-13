import torch
import numpy as np
from .utils import np_groupby

from ase.neighborlist import neighbor_list


def atoms2graph(atoms, cutoff, edge_dist=False):
    """Convert an ASE `Atoms` object into a graph based on a radius cutoff.
    Returns the graph (in COO format),
    its node attributes (atom types, `x`),
    and its edge attributes (format determined by `edge_dist`).

    Args:
        atoms (ase.Atoms): Collection of atoms to be converted to a graph.
        cutoff (float): Cutoff radius for nearest neighbor search.
        edge_dist (bool, optional): Set to `True` to output edge distances.
            Otherwise, output edge vectors.

    Returns:
       tuple: Tuple of (edge_index, x, edge_attr) that describes the atomic graph.

    :rtype: (ndarray, ndarray, ndarray)
    """
    _, x = np.unique(atoms.numbers, return_inverse=True)
    i, j, d, D = neighbor_list('ijdD', atoms, cutoff)
    if edge_dist:
        return np.stack((i, j)), x, d.astype(np.float32)
    else:
        return np.stack((i, j)), x, D.astype(np.float32)


def atoms2knngraph(atoms, cutoff, k=12, scale_inv=True):
    """Convert an ASE `Atoms` object into a graph based on k nearest neighbors.
    Returns the graph (in COO format),
    its node attributes (atom types `x`),
    and its edge attributes (distance vectors `edge_attr`).

    Args:
        atoms (ase.Atoms): Collection of atoms to be converted to a graph.
        cutoff (float): Cutoff radius for nearest neighbor search.
            These neighbors are then down-selected to k nearest neighbors.
        k (int, optional): Number of nearest neighbors for each atom.
        scale_inv (bool, optional): If set to `True`, normalize the distance
            vectors `edge_attr` such that each atom's furthest neighbor is
            one unit distance away. This makes the knn graph scale-invariant.

    Returns:
       tuple: Tuple of (edge_index, x, edge_attr) that describes the knn graph.

    :rtype: (ndarray, ndarray, ndarray)
    """
    edge_src, edge_dst, edge_dists, edge_vecs = neighbor_list('ijdD', atoms, cutoff=cutoff)

    src_groups  = np_groupby(edge_src, groups=edge_dst)
    dst_groups  = np_groupby(edge_dst, groups=edge_dst)
    dist_groups = np_groupby(edge_dists, groups=edge_dst)
    vec_groups  = np_groupby(edge_vecs, groups=edge_dst)

    knn_idx = [np.argsort(d)[:k] for d in dist_groups]
    for indices in knn_idx:
        if len(indices) != k:
            raise Exception("The number of nearest neighbors is not K. Consider increasing the cutoff radius.")

    src_knn = tuple(s[indices] for s, indices in zip(src_groups, knn_idx))
    dst_knn = tuple(d[indices] for d, indices in zip(dst_groups, knn_idx))
    vec_knn = tuple(v[indices] for v, indices in zip(vec_groups, knn_idx))

    if scale_inv:
        vec_knn = [v / np.linalg.norm(v, axis=1).max() for v in vec_knn]

    i = np.concatenate(src_knn)
    j = np.concatenate(dst_knn)
    D = np.concatenate(vec_knn)

    edge_index = np.stack((i, j))
    _, x = np.unique(atoms.numbers, return_inverse=True)
    edge_attr = D.astype(np.float32)
    return edge_index, x, edge_attr