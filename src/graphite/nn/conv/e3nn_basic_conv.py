import torch
from torch_geometric.utils import scatter

from e3nn      import o3
from e3nn.nn   import FullyConnectedNet


class BasicConv(torch.nn.Module):
    """Equivariant basic convolution.
    Reference: https://docs.e3nn.org/en/stable/guide/convolution.html.

    Args:
        irreps_in (Irreps or str): Irreps of input node features.
        irreps_out (Irreps or str): Irreps of output node features.
        irreps_sh (Irreps or str): Irreps of edge spherical harmonics (constant througout model).
        radial_neurons (list of ints): Number of neurons per layers in the radial MLP.
            For first and hidden layers, not the output layer.
        num_neighbors (float): Typical or averaged node degree (used for normalization).
    """
    def __init__(self,
        irreps_in,
        irreps_out,
        irreps_sh,
        radial_neurons = [16, 64],
        num_neighbors  = 1,
    ):
        super().__init__()

        self.irreps_in     = o3.Irreps(irreps_in)
        self.irreps_out    = o3.Irreps(irreps_out)
        self.irreps_sh     = o3.Irreps(irreps_sh)
        self.num_neighbors = num_neighbors

        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            shared_weights=False,
        )

        self.mlp = FullyConnectedNet(radial_neurons + [self.tp.weight_numel], torch.nn.functional.silu)

    def forward(self, x, edge_index, edge_attr, edge_len_emb):
        i, j = edge_index
        num_nodes = len(x)
        summand = self.tp(x[i], edge_attr, self.mlp(edge_len_emb))
        return scatter(summand, j, dim=0, dim_size=num_nodes).div(self.num_neighbors**0.5)
