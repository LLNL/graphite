import torch
from torch.nn   import Embedding, ModuleList

from e3nn       import o3
from e3nn.math  import soft_one_hot_linspace
from e3nn.nn    import Activation
from ..conv.e3nn_mgn_conv import MGNConv


class Compose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, *input):
        x, y = self.first(*input)
        return self.second(x), self.second(y)


class MGNConvNet(torch.nn.Module):
    """Equivariant version of MGN model

    Args:
        irreps_in (Irreps or str): Irreps of input node features. Must be scalars.
        irreps_node (Irreps or str): Irreps of node attributes (not updated throughout model). Must be scalars.
        irreps_hidden (Irreps or str): Irreps of node features at hidden layers.
        irreps_edge (Irreps or str): Irreps of spherical_harmonics.
        irreps_out (Irreps or str): Irreps of output node features.
        num_convs (int): Number of interaction/conv layers. Must be more than 1.
        radial_neurons (list of ints): Number neurons per layers in the FC network that learn from bond distances.
            For first and hidden layers, not the output layer.
        num_species (int): Number of elements/species in the atomic data.
        max_radius (float): Cutoff radius used during graph construction.
        num_neighbors (float): Typical or average node degree (used for normalization).
    """
    def __init__(self,
        irreps_node_in  = '8x0e',
        irreps_edge_in  = '1x0e + 1x1e + 1x2e',
        irreps_hidden   = '8x0e + 8x1e + 8x2e',
        irreps_out      = '8x0e',
        num_convs       = 3,
        radial_neurons  = [16, 64],
        num_species     = 1,
        max_radius      = 3.0,
    ):
        super().__init__()

        self.irreps_node_in  = o3.Irreps(irreps_node_in)
        self.irreps_edge_in  = o3.Irreps(irreps_edge_in)
        self.irreps_hidden   = o3.Irreps(irreps_hidden)
        self.irreps_out      = o3.Irreps(irreps_out)
        self.num_convs       = num_convs
        self.max_radius      = max_radius
        self.num_edge_basis  = radial_neurons[0]
        self.embed_node      = Embedding(num_species, self.irreps_node_in.dim)

        act_dict = {1: torch.nn.functional.silu, -1: torch.tanh}
        acts = [act_dict[ir.p] if ir[0] == 0 else None for _, ir in self.irreps_hidden]

        self.convs = ModuleList()

        conv = MGNConv(
            irreps_node_in  = self.irreps_node_in,
            irreps_node_out = self.irreps_hidden,
            irreps_edge_in  = self.irreps_edge_in,
            irreps_edge_out = self.irreps_hidden,
            radial_neurons  = radial_neurons,
        )
        act = Activation(self.irreps_hidden, acts)
        self.convs.append(Compose(conv, act))

        for l in range(num_convs-1):
            conv = MGNConv(
                irreps_node_in  = self.irreps_hidden,
                irreps_node_out = self.irreps_hidden,
                irreps_edge_in  = self.irreps_hidden,
                irreps_edge_out = self.irreps_hidden,
                radial_neurons  = radial_neurons,
            )
            act = Activation(self.irreps_hidden, acts)
            self.convs.append(Compose(conv, act))

        self.out = o3.FullyConnectedTensorProduct(
            irreps_in1 = self.irreps_hidden,
            irreps_in2 = self.irreps_hidden,
            irreps_out = self.irreps_out,
        )

    def forward(self, data):
        x, edge_index, edge_vec = data.x, data.edge_index, data.edge_attr

        h_atm = self.embed_node(x)

        edge_len_emb = soft_one_hot_linspace(
            edge_vec.norm(dim=1),
            start  = 0.0,
            end    = self.max_radius,
            number = self.num_edge_basis,
            basis  = 'smooth_finite',
            cutoff = True,
        )
  
        h_bnd = o3.spherical_harmonics(self.irreps_edge_in, edge_vec, normalize=True, normalization='component')

        for conv in self.convs:
            h_atm, h_bnd = conv(h_atm, edge_index, h_bnd, edge_len_emb)

        return self.out(h_atm, h_atm)
