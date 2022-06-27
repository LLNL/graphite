import torch
from torch.nn   import Embedding, ModuleList

from e3nn       import o3
from e3nn.math  import soft_one_hot_linspace
from e3nn.nn    import FullyConnectedNet
from ..conv.e3nn_basic_conv import E3NN_BasicConv


class E3NN_PhaseNet_simple(torch.nn.Module):
    """Phase characterization model consisted of e3nn's `Convolution` layers, followed by MLP.
    The `Convolution` operation is described at https://docs.e3nn.org/en/stable/guide/convolution.html.
    The MLP is not equivariant. Therefore the irreps of latent features after the Convolution layers should be scalar only.

    Args:
        irreps_in (e3nn.o3.Irreps or str): Irreps of input node features.
        irreps_hidden (e3nn.o3.Irreps or str): Irreps of node features at hidden layers.
        irreps_emb (e3nn.o3.Irreps or str): Irreps of embedding after convolutions. Must be scalars.
        irreps_edge (e3nn.o3.Irreps or str): Irreps of spherical_harmonics.
        num_convs (int): Number of interaction/conv layers. Must be more than 1.
        num_species (int): Number of elements/species in the atomic data.
        num_neighbors (float): Typical or average node degree (used for normalization).
        max_radius (float): Cutoff radius used during graph construction.
        radial_neurons (list of ints): Number neurons per layers in the FC network that learns from bond distances.
            For first and hidden layers, not the output layer.
        head_neurons (list of ints): Number of neurons per layers in the FC network that projects to final output.
            For hidden and last layers, not the first layer.
    """
    def __init__(self,
        irreps_in      = '4x0e',
        irreps_hidden  = '8x4e + 8x6e',
        irreps_emb     = '4x0e',
        irreps_edge    = '1x4e + 1x6e',
        num_convs      = 3,
        num_species    = 1,
        num_neighbors  = 12,
        max_radius     = 3.15,
        radial_neurons = [16, 64],
        head_neurons   = [64, 4],
    ):
        super().__init__()

        self.irreps_in      = o3.Irreps(irreps_in)
        self.irreps_hidden  = o3.Irreps(irreps_hidden)
        self.irreps_emb     = o3.Irreps(irreps_emb)
        self.irreps_edge    = o3.Irreps(irreps_edge)
        self.num_convs      = num_convs
        self.max_radius     = max_radius
        self.num_edge_basis = radial_neurons[0]
        self.node_embedding = Embedding(num_species, self.irreps_in[0][0])

        self.tp_convs = ModuleList()
        for l in range(num_convs-1):
            self.tp_convs.append(
                E3NN_BasicConv(
                    self.irreps_in if l == 0 else irreps_hidden,
                    self.irreps_hidden,
                    irreps_sh      = irreps_edge,
                    radial_neurons = radial_neurons,
                    num_neighbors  = num_neighbors,
                )
            )

        self.tp_convs.append(
            E3NN_BasicConv(
                self.irreps_hidden,
                self.irreps_emb,
                irreps_sh      = irreps_edge,
                radial_neurons = radial_neurons,
                num_neighbors  = num_neighbors,
            )
        )

        self.head = FullyConnectedNet([self.irreps_emb[0][0]] + head_neurons, torch.nn.functional.silu)

    def forward(self, data):
        x, edge_index, edge_vec = data.x, data.edge_index, data.edge_attr

        h_atm = self.node_embedding(x)

        h_bnd = soft_one_hot_linspace(
            edge_vec.norm(dim=1),
            start  = 0.0,
            end    = self.max_radius,
            number = self.num_edge_basis,
            basis  = 'smooth_finite',
            cutoff = True,
        ).mul(self.num_edge_basis**0.5)

        sh = o3.spherical_harmonics(
            self.irreps_edge,
            edge_vec,
            normalize     = True,
            normalization = 'component',
        )

        for conv in self.tp_convs:
            h_atm = conv(h_atm, edge_index, sh, h_bnd)

        if self.training:
            return self.head(h_atm)
        else:
            return self.head(h_atm), h_atm
