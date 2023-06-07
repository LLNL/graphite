import torch
from torch_geometric.utils import scatter

from e3nn      import o3
from e3nn.nn   import FullyConnectedNet
from e3nn.math import soft_unit_step

class SelfAttention(torch.nn.Module):
    """Equivariant self-attention mechanism.
    References:
    * https://proceedings.neurips.cc/paper/2020/file/15231a7ce4ba789d13b722cc5c955834-Paper.pdf
    * https://docs.e3nn.org/en/stable/guide/transformer.html

    Args:
        irreps_in (Irreps or str): Irreps of input node features.
        irreps_q (Irreps or str): Irreps of query features
        irreps_k (Irreps or str): Irreps of key features
        irreps_v (Irreps or str): Irreps of value features (also the irreps of output node features).
        irreps_sh (Irreps or str): Irreps of edge spherical harmonics.
        max_radius (float): Radius cutoff used during graph construction.
        radial_neurons (list of ints): Number of neurons per layers in the MLPs for creating the queries and keys.
            For first and hidden layers, not the output layer.
        num_neighbors (float): Typical or averaged node degree (used for normalization).
    """
    def __init__(self,
        irreps_in,
        irreps_q,
        irreps_k,
        irreps_v,
        irreps_sh      = o3.Irreps.spherical_harmonics(2),
        max_radius     = 3.15,
        radial_neurons = [16, 64],
        num_neighbors  = 1,
    ):
        super().__init__()

        self.irreps_in      = o3.Irreps(irreps_in)
        self.irreps_q       = o3.Irreps(irreps_q)
        self.irreps_k       = o3.Irreps(irreps_k)
        self.irreps_v       = o3.Irreps(irreps_v)
        self.irreps_sh      = o3.Irreps(irreps_sh)
        self.max_radius     = max_radius
        self.num_edge_basis = radial_neurons[0]
        self.num_neighbors  = num_neighbors

        # Query
        self.h_q = o3.Linear(irreps_in, irreps_q)

        # Key
        self.tp_k = o3.FullyConnectedTensorProduct(irreps_in, irreps_sh, irreps_k, shared_weights=False)
        self.fc_k = FullyConnectedNet(radial_neurons + [tp_k.weight_numel], torch.nn.functional.silu)

        # Value
        self.tp_v = o3.FullyConnectedTensorProduct(irreps_in, irreps_sh, irreps_v, shared_weights=False)
        self.fc_v = FullyConnectedNet(radial_neurons + [tp_v.weight_numel], torch.nn.functional.silu)

        # Dot product of q and k
        self.dot = o3.FullyConnectedTensorProduct(irreps_q, irreps_k, '0e')

    def forward(self, x, edge_index, edge_attr, edge_len_embbed):
        i, j = edge_index

        # Compute queries (per node), keys (per edge), and values (per edge).
        q = self.h_q(x)
        k = self.tp_k(x[i], edge_attr, self.fc_k(edge_len_embbed))
        v = self.tp_v(x[i], edge_attr, self.fc_v(edge_len_embbed))

        # Compute attention weights (per edge)
        edge_weight_cutoff = soft_unit_step(10 * (1 - edge_len / self.max_radius))
        exp = edge_weight_cutoff[:, None] * self.dot(q[j], k).exp()  # Numerator
        Z   = scatter(exp, j, dim=0, dim_size=len(x))  # Denominator
        alpha = exp / Z[j]

        # Return weighted sums of the values.
        return scatter(alpha.relu().sqrt() * v, j, dim=0, dim_size=len(x))
