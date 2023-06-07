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
