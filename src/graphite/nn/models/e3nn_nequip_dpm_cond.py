import torch
import torch.nn as nn

from e3nn       import o3
from e3nn.nn    import Gate
from e3nn.math  import soft_one_hot_linspace

from ..conv.e3nn_nequip_interaction import Interaction
from ..basis import GaussianFourierProjection


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class Compose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)


class NequIP_dpm_cond(torch.nn.Module):
    """NequIP model from https://arxiv.org/pdf/2101.03164.pdf.

    Args:
        irreps_in (Irreps or str): Irreps of input node features.
        irreps_node (Irreps or str): Irreps of node attributes (not updated throughout model). Must be scalars.
        irreps_hidden (Irreps or str): Irreps of node features at hidden layers.
        irreps_edge (Irreps or str): Irreps of spherical_harmonics.
        irreps_out (Irreps or str): Irreps of output node features.
        num_convs (int): Number of interaction/conv layers. Must be more than 1.
        radial_neurons (list of ints): Number of neurons per layers in the FC network that learn from bond distances.
            For first and hidden layers, not the output layer.
        num_species (int): Number of elements/species in the atomic data.
        max_radius (float): Cutoff radius used during graph construction.
        num_neighbors (float): Typical or average node degree (used for normalization).
    """
    def __init__(self,
        irreps_in      = '16x0e + 1x1e',
        irreps_node    = '64x0e',
        irreps_hidden  = '64x0e + 32x1e + 32x2e',
        irreps_edge    = '1x0e + 1x1e + 1x2e',
        irreps_out     = '1x1e',
        num_convs      = 3,
        radial_neurons = [32, 64],
        num_species    = 1,
        max_radius     = 5.0,
        num_neighbors  = 12,
        embed_t_dim    = 128,
    ):
        super().__init__()

        self.irreps_in      = o3.Irreps(irreps_in)
        self.irreps_hidden  = o3.Irreps(irreps_hidden)
        self.irreps_out     = o3.Irreps(irreps_out)
        self.irreps_node    = o3.Irreps(irreps_node)
        self.irreps_edge    = o3.Irreps(irreps_edge)
        self.num_convs      = num_convs
        self.max_radius     = max_radius
        self.num_edge_basis = radial_neurons[0]
        self.embed_node_x   = nn.Embedding(num_species, self.irreps_in[0].dim)
        self.embed_node_z   = nn.Embedding(num_species, self.irreps_node.dim)
        self.embed_time     = nn.Sequential(
            GaussianFourierProjection(embed_t_dim),
            nn.Linear(embed_t_dim, embed_t_dim),
            nn.SiLU(),
            nn.Linear(embed_t_dim, self.irreps_node.dim)
        )
        # self.embed_bnd_type = nn.Embedding(2, self.num_edge_basis//2)

        act_scalars = {1: torch.nn.functional.silu, -1: torch.tanh}
        act_gates   = {1: torch.sigmoid, -1: torch.tanh}

        irreps = self.irreps_in
        self.interactions = nn.ModuleList()
        for _ in range(num_convs):
            irreps_scalars = o3.Irreps([(m, ir) for m, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge, ir)])
            irreps_gated   = o3.Irreps([(m, ir) for m, ir in self.irreps_hidden if ir.l > 0  and tp_path_exists(irreps, self.irreps_edge, ir)])

            if irreps_gated.dim > 0:
                if tp_path_exists(irreps_node, self.irreps_edge, "0e"):
                    ir = "0e"
                elif tp_path_exists(irreps_node, self.irreps_edge, "0o"):
                    ir = "0o"
                else:
                    raise ValueError(f"irreps={irreps} times irreps_edge={self.irreps_edge} is unable to produce gates needed for irreps_gated={irreps_gated}.")
            else:
                ir = None
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            gate = Gate(
                irreps_scalars, [act_scalars[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,   [act_gates[ir.p]   for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )

            conv = Interaction(
                irreps_in      = irreps,
                irreps_node    = self.irreps_node,
                irreps_edge    = self.irreps_edge,
                irreps_out     = gate.irreps_in,
                radial_neurons = radial_neurons,
                num_neighbors  = num_neighbors,
            )
            irreps = gate.irreps_out
            self.interactions.append(Compose(conv, gate))

        self.out = o3.FullyConnectedTensorProduct(
            irreps_in1 = irreps,
            irreps_in2 = self.irreps_node,
            irreps_out = self.irreps_out,
        )

    def forward(self, x, x_cond, t, sigma=1.0):
        """Predict noise/score given a noised input `x`, a conditional input `data`, and schedule time `t`.

        Args:
            x: The variable to be de-noised, e.g., (conditional) atom displacements.
            x_cond: The conditional particle data as a PyG `Data` graph object.
            t: Time w.r.t. noise schedule.
        """
        data = x_cond

        x_atm, edge_index, edge_vec = data.x, data.edge_index, data.edge_attr

        z     = self.embed_node_z(x_atm) + self.embed_time(t)[data.batch]
        h_atm = torch.cat([self.embed_node_x(x_atm), x], dim=-1)

        h_bnd = soft_one_hot_linspace(
            edge_vec.norm(dim=1),
            start  = 0.0,
            end    = self.max_radius,
            number = self.num_edge_basis,
            basis  = 'smooth_finite',
            cutoff = True,
        ).mul(self.num_edge_basis**0.5)

        # h_bnd_type = self.embed_bnd_type(data.mask_real_bnd.to(torch.long))
        # h_bnd = torch.cat([h_bnd_len, h_bnd_type], dim=-1)
        # h_bnd = h_bnd_len

        edge_sh = o3.spherical_harmonics(self.irreps_edge, edge_vec, normalize=True, normalization='component')

        for layer in self.interactions:
            h_atm = layer(h_atm, z, edge_index, edge_sh, h_bnd)

        return self.out(h_atm, z) / sigma
