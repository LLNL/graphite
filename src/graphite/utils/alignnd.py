import torch
import numpy as np
import itertools
from functools import partial
from .utils import np_scatter

from ase.neighborlist import neighbor_list


permute_2 = partial(itertools.permutations, r=2)
def line_graph(edge_index_G):
    """Return the (angular) line graph of the input graph.

    Args:
        edge_index_G (ndarray): Input graph in COO format.
    """
    src_G, dst_G = edge_index_G
    edge_index_A = [
        (u, v)
        for edge_pairs in np_scatter(np.arange(len(dst_G)), dst_G, permute_2)
        for u, v in edge_pairs
    ]
    return np.array(edge_index_A).T


def dihedral_graph(edge_index_G):
    """Return the "dihedral angle line graph" of the input graph.

    Args:
        edge_index_G (ndarray): Input graph in COO format.
    """
    src, dst = edge_index_G
    edge_index_A = [
        (u, v)
        for i, j in edge_index_G.T
        for u in np.flatnonzero((dst == i) & (src != j))
        for v in np.flatnonzero((dst == j) & (src != i))
    ]
    return np.array(edge_index_A).T


def get_bnd_angs(atoms, edge_index_G, edge_index_A_bnd_ang):
    """Return the bond angles (in radians) for the (angular) line graph edges.
    """
    indices  = edge_index_G.T[edge_index_A_bnd_ang.T].reshape(-1, 4)
    bnd_angs = atoms.get_angles(indices[:, [0, 1, 2]])
    return np.radians(bnd_angs)


def get_dih_angs(atoms, edge_index_G, edge_index_A_dih_ang):
    """Return the dihedral angles (in radians) for the dihedral line graph edges.
    """
    indices  = edge_index_G.T[edge_index_A_dih_ang.T].reshape(-1, 4)
    dih_angs = atoms.get_dihedrals(indices[:, [0, 1, 3, 2]])
    return np.radians(dih_angs)
