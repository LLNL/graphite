import torch
import torch.nn as nn

from e3nn       import o3
from e3nn.nn    import Gate

from ..conv.e3nn_nequip import Interaction


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class Compose(nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)


class NequIP(nn.Module):
    """NequIP model from https://arxiv.org/pdf/2101.03164.pdf.

    Args:
        init_embed (function): Initial embedding function/class for nodes and edges.
        irreps_node_x (Irreps or str): Irreps of input node features.
        irreps_node_z (Irreps or str): Irreps of auxiliary node features (not updated throughout model).
        irreps_hidden (Irreps or str): Irreps of node features at hidden layers.
        irreps_edge (Irreps or str): Irreps of spherical_harmonics.
        irreps_out (Irreps or str): Irreps of output node features.
        num_convs (int): Number of interaction/conv layers. Must be more than 1.
        radial_neurons (list of ints): Number of neurons per layers in the MLP that learns from bond distances.
            For first and hidden layers, not the output layer.
        max_radius (float): Cutoff radius used during graph construction.
        num_neighbors (float): Typical or average node degree (used for normalization).
    
    Notes:
        The `init_embed` function/class must take a PyG graph object `data` as input and output the same object
        with the additional fields `h_node_x`, `h_node_z`, and `h_edge` that correspond to the node, auxilliary node,
        and edge embeddings.
    """
    def __init__(self,
        init_embed,
        irreps_node_x  = '8x0e',
        irreps_node_z  = '8x0e',
        irreps_hidden  = '64x0e + 32x1e + 32x2e',
        irreps_edge    = '1x0e + 1x1e + 1x2e',
        irreps_out     = '1x1e',
        num_convs      = 3,
        radial_neurons = [16, 64],
        num_neighbors  = 12,
    ):
        super().__init__()
        self.init_embed     = init_embed
        self.irreps_node_x  = o3.Irreps(irreps_node_x)
        self.irreps_node_z  = o3.Irreps(irreps_node_z)
        self.irreps_hidden  = o3.Irreps(irreps_hidden)
        self.irreps_out     = o3.Irreps(irreps_out)
        self.irreps_edge    = o3.Irreps(irreps_edge)
        self.num_convs      = num_convs

        act_scalars = {1: nn.functional.silu, -1: torch.tanh}
        act_gates   = {1: torch.sigmoid, -1: torch.tanh}

        irreps = self.irreps_node_x
        self.interactions = nn.ModuleList()
        for _ in range(num_convs):
            irreps_scalars = o3.Irreps([(m, ir) for m, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge, ir)])
            irreps_gated   = o3.Irreps([(m, ir) for m, ir in self.irreps_hidden if ir.l > 0  and tp_path_exists(irreps, self.irreps_edge, ir)])

            if irreps_gated.dim > 0:
                if tp_path_exists(irreps_node_z, self.irreps_edge, "0e"):
                    ir = "0e"
                elif tp_path_exists(irreps_node_z, self.irreps_edge, "0o"):
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
                irreps_node    = self.irreps_node_z,
                irreps_edge    = self.irreps_edge,
                irreps_out     = gate.irreps_in,
                radial_neurons = radial_neurons,
                num_neighbors  = num_neighbors,
            )
            irreps = gate.irreps_out
            self.interactions.append(Compose(conv, gate))

        self.out = o3.FullyConnectedTensorProduct(
            irreps_in1 = irreps,
            irreps_in2 = self.irreps_node_z,
            irreps_out = self.irreps_out,
        )

    def forward(self, data):
        # Embedding
        data = self.init_embed(data)
        edge_index, edge_attr = data.edge_index, data.edge_attr
        h_node_x, h_node_z, h_edge = data.h_node_x, data.h_node_z, data.h_edge

        # Graph convolutions
        edge_sh = o3.spherical_harmonics(self.irreps_edge, edge_attr, normalize=True, normalization='component')
        for layer in self.interactions:
            h_node_x = layer(h_node_x, h_node_z, edge_index, edge_sh, h_edge)

        # Final output layer
        return self.out(h_node_x, h_node_z)
