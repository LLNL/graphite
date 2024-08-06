"""Conventional order parameter featurization
Reference: https://freud.readthedocs.io/en/latest/modules/order.html
"""

import torch
from torch_geometric.utils import scatter

from e3nn import o3


def steinhardt(l, edge_index, edge_vec, num_nodes, p=1, second_shell_avg=True):
    # Compute q_lm
    i, j = edge_index
    irreps_sh = o3.Irreps([(1, (l, p))])
    sh = o3.spherical_harmonics(irreps_sh, edge_vec, normalize=True, normalization='norm')
    q_lm = scatter(sh, index=j, dim=0, reduce='mean', dim_size=num_nodes)

    # If performing a second averaging to include second shell neighbors
    if second_shell_avg:
        q_lm = scatter(q_lm[i], index=j, dim=0, reduce='mean', dim_size=num_nodes)

    q_lm_square_sum = q_lm.abs().pow(2).sum(1)

    # Compute w_l
    w_l = torch.einsum('li, lj, lk, ijk -> l', q_lm, q_lm, q_lm, o3.wigner_3j(l,l,l).to(q_lm.device))
    w_l = w_l / q_lm_square_sum.pow(3/2)

    # Compute q_l
    q_l = 4*torch.pi / (2*l + 1) * q_lm_square_sum
    q_l = q_l.sqrt()
    return q_l, w_l

# from torch_scatter import scatter
"""
If cutoff=infinity and return_qlm=False, it should behave exactly as steinhardt() above

Weighted Steinhardt order parameters
Inspired by W. Mickel, S. C. Kapfer, G. E. Schröder-Turk, and K. Mecke, 
  Shortcomings of the Bond Orientational Order Parameters for the Analysis of Disordered Particulate Matter,
  J. Chem. Phys. 138, 044501 (2013).
But to simplify implementation, I am simply applying weights as a function of bond length!!!

"""
def steinhardt_weighted(l, edge_index, edge_vec, num_nodes, p=1, second_shell_avg=True, cutoff=1.6, decay_len=0.2, return_qlm=False):
    # Compute q_lm
    i, j = edge_index
    irreps_sh = o3.Irreps([(1, (l, p))])
    edge_len = edge_vec.norm(dim=-1, keepdim=True)
    weight = torch.exp(-(torch.nn.functional.relu(edge_len-cutoff)/decay_len)**2)
    sh = o3.spherical_harmonics(irreps_sh, edge_vec, normalize=True, normalization='norm')
    sh = sh*weight
    # q_lm = scatter(sh, j, dim=0, reduce='mean')
    q_lm = scatter(sh, index=j, dim=0, reduce='mean', dim_size=num_nodes)

    # If performing a second averaging to include second shell neighbors
    if second_shell_avg:
        # q_lm = scatter(q_lm[i], j, dim=0, reduce='mean')
        q_lm = scatter(q_lm[i], index=j, dim=0, reduce='mean', dim_size=num_nodes)
    if return_qlm: return q_lm

    q_lm_square_sum = q_lm.abs().pow(2).sum(1)

    # Compute w_l
    w_l = torch.einsum('li, lj, lk, ijk -> l', q_lm, q_lm, q_lm, o3.wigner_3j(l,l,l).to(q_lm.device))
    w_l = w_l / q_lm_square_sum.pow(3/2)

    # Compute q_l
    q_l = 4*torch.pi / (2*l + 1) * q_lm_square_sum
    q_l = q_l.sqrt()
    return q_l, w_l
