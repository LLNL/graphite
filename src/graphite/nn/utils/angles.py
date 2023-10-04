import torch
from torch.linalg import cross


def bond_angles(bond_vec, edge_index_bnd_ang):
    bond_vec /= torch.linalg.norm(bond_vec, dim=-1, keepdim=True)
    i = edge_index_bnd_ang[0]
    j = edge_index_bnd_ang[1]
    cos_ang = (bond_vec[i] * bond_vec[j]).sum(dim=-1, keepdim=True).clamp(min=-1., max=1.)
    sin_ang = cos_ang.acos().sin()
    return torch.hstack([cos_ang, sin_ang])


def dihedral_angles(pos, edge_index_bnd, edge_index_dih_ang):
    """Does not account for periodic boundaries"""
    dih_idx = edge_index_bnd.T[edge_index_dih_ang.T].reshape(-1, 4)
    dih_idx = dih_idx.T
    i, j, k, l = dih_idx[0], dih_idx[1], dih_idx[3], dih_idx[2]
    u1 = pos[j] - pos[i]
    u2 = pos[k] - pos[j]
    u3 = pos[l] - pos[k]
    u1 /= torch.linalg.norm(u1, dim=-1, keepdim=True)
    u2 /= torch.linalg.norm(u2, dim=-1, keepdim=True)
    u3 /= torch.linalg.norm(u3, dim=-1, keepdim=True)
    cos_ang = (cross(u1, u2) * cross(u2, u3)).sum(dim=-1, keepdim=True).clamp(min=-1., max=1.)
    sin_ang = (u1 * cross(u2, u3)).sum(dim=-1, keepdim=True).clamp(min=-1., max=1.)
    return torch.hstack([cos_ang, sin_ang])