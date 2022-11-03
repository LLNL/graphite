import torch
from torch.nn   import Embedding, ModuleList

from e3nn       import o3
from e3nn.math  import soft_one_hot_linspace
from ..conv.e3nn_basic_conv import BasicConv


class SimpleConvNet(torch.nn.Module):
    """Simple e3nn model consisted of the `Convolution` operation documented at
    https://docs.e3nn.org/en/stable/guide/convolution.html.

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
        irreps_in      = '4x0e',
        irreps_node    = '4x0e',
        irreps_hidden  = '64x0e + 16x1e',
        irreps_edge    = '1x0e + 1x1e + 1x2e',
        irreps_out     = '4x0e',
        num_convs      = 3,
        radial_neurons = [16, 64],
        num_species    = 1,
        max_radius     = 3.15,
        num_neighbors  = 12,
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
        self.node_embedding = Embedding(num_species, self.irreps_in.dim)

        self.tp_convs = ModuleList()
        for l in range(num_convs):
            self.tp_convs.append(
                BasicConv(
                    self.irreps_in if l == 0 else irreps_hidden,
                    self.irreps_hidden,
                    irreps_sh      = irreps_edge,
                    radial_neurons = radial_neurons,
                    num_neighbors  = num_neighbors,
                )
            )

        self.out = o3.FullyConnectedTensorProduct(
            irreps_in1 = self.irreps_hidden,
            irreps_in2 = self.irreps_node,
            irreps_out = self.irreps_out,
        )

    def forward(self, data):
        x, edge_index, edge_vec = data.x, data.edge_index, data.edge_attr

        z = self.embed_node_z(x)
        h_atm = self.node_embedding(x)

        h_bnd = soft_one_hot_linspace(
            edge_vec.norm(dim=1),
            start  = 0.0,
            end    = self.max_radius,
            number = self.num_edge_basis,
            basis  = 'smooth_finite',
            cutoff = True,
        ).mul(self.num_edge_basis**0.5)
  
        edge_sh = o3.spherical_harmonics(self.irreps_edge, edge_vec, normalize=True, normalization='component')

        for conv in self.tp_convs:
            h_atm = conv(h_atm, edge_index, edge_sh, h_bnd)

        return self.out(h_atm, z)
