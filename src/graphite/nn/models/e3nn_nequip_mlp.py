import torch

from e3nn       import o3
from e3nn.nn    import FullyConnectedNet
from .e3nn_nequip import NequIP


class NequIP_MLP(torch.nn.Module):
    """Phase characterization model consisted of the `Interaction` layers, followed by MLP.
    The `Interaction` layer is based on the NequIP model from https://arxiv.org/pdf/2101.03164.pdf.
    The MLP is not equivariant. Therefore the irreps of latent features after the Interactions layers should be scalar only.

    Args:
        irreps_in (Irreps or str): Irreps of input node features. Must be scalars.
        irreps_node (Irreps or str): Irreps of node attributes (not updated throughout model). Must be scalars.
        irreps_hidden (Irreps or str): Irreps of node features at hidden layers.
        irreps_edge (Irreps or str): Irreps of spherical_harmonics.
        irreps_emb (Irreps or str): Irreps of embedding after convolutions/interactions. Must be scalars.
        num_convs (int): Number of interaction/conv layers. Must be more than 1.
        num_species (int): Number of elements/species in the atomic data.
        num_neighbors (float): Typical or average node degree (used for normalization).
        max_radius (float): Cutoff radius used during graph construction.
        radial_neurons (list of ints): Number neurons per layers in the FC network that learn from bond distances.
            For first and hidden layers, not the output layer.
        head_neurons (list of ints): Number of neurons per layers in the FC network that projects to final output.
            For hidden and last layers, not the first layer.
    """
    def __init__(self,
        irreps_in      = '8x0e',
        irreps_node    = '8x0e',
        irreps_hidden  = '8x0e + 8x1e + 8x2e',
        irreps_edge    = '1x0e + 1x1e + 1x2e',
        irreps_emb     = '64x0e',
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
        self.irreps_node    = o3.Irreps(irreps_node)
        self.irreps_edge    = o3.Irreps(irreps_edge)

        self.nequip_model = NequIP(
            irreps_in      = self.irreps_in,
            irreps_hidden  = self.irreps_hidden,
            irreps_out     = self.irreps_emb,
            irreps_node    = self.irreps_node,
            irreps_edge    = self.irreps_edge,
            num_convs      = num_convs,
            radial_neurons = radial_neurons,
            num_species    = num_species,
            max_radius     = max_radius,
            num_neighbors  = num_neighbors,
        )

        self.head = FullyConnectedNet([self.irreps_emb.dim] + head_neurons, torch.nn.functional.silu)

    def forward(self, data):
        h_atm = self.nequip_model(data)
        return self.head(h_atm)
