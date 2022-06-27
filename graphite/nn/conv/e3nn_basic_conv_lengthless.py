import torch
from torch_scatter import scatter

from e3nn      import o3


class E3NN_BasicConv_lengthless(torch.nn.Module):
    """Equivariant basic convolution.
    Reference: https://docs.e3nn.org/en/stable/guide/convolution.html.

    Args:
        irreps_in (e3nn.o3.Irreps or str): Irreps of input node features.
        irreps_out (e3nn.o3.Irreps or str): Irreps of output node features.
        irreps_sh (e3nn.o3.Irreps or str): Irreps of edge spherical harmonics (constant througout model).
        num_neighbors (float): Typical or averaged node degree (used for normalization).
    """
    def __init__(self,
        irreps_in,
        irreps_out,
        irreps_sh,
        num_neighbors = 1,
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
        )

    def forward(self, x, edge_index, edge_attr):
        i, j = edge_index
        num_nodes = len(x)
        summand = self.tp(x[i], edge_attr)
        return scatter(summand, j, dim=0, dim_size=num_nodes).div(self.num_neighbors**0.5)
