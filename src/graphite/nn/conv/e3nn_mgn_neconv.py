import torch
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from e3nn      import o3
from e3nn.nn   import FullyConnectedNet


class MGNConv(torch.nn.Module):
    """Equivariant MGN convolution.

    Args:
        irreps_in (Irreps or str): Irreps of input node features.
        irreps_out (Irreps or str): Irreps of output node features.
        irreps_sh (Irreps or str): Irreps of edge spherical harmonics (constant througout model).
        radial_neurons (list of ints): Number of neurons per layers in the radial MLP.
            For first and hidden layers, not the output layer.
    """
    def __init__(self,
        irreps_node_in,
        irreps_node_out,
        irreps_edge_in,
        irreps_edge_out,
        radial_neurons = [16, 64],
    ):
        super().__init__()

        self.irreps_node_in  = o3.Irreps(irreps_node_in)
        self.irreps_node_out = o3.Irreps(irreps_node_out)
        self.irreps_edge_in  = o3.Irreps(irreps_edge_in)
        self.irreps_edge_out = o3.Irreps(irreps_edge_out)

        self.edge_tp = o3.FullyConnectedTensorProduct(
            self.irreps_edge_in,
            self.irreps_node_in*2,
            self.irreps_edge_out,
            internal_weights = False,
            shared_weights   = False,
        )

        self.node_tp = o3.FullyConnectedTensorProduct(
            self.irreps_node_in,
            self.irreps_edge_out,
            self.irreps_node_out,
        )

        self.mlp = FullyConnectedNet(radial_neurons + [self.edge_tp.weight_numel], torch.nn.functional.silu)

    def forward(self, x, edge_index, edge_attr, edge_len_emb):
        i, j = edge_index

        # Update the edges
        z = torch.cat([x[i], x[j]], dim=-1)
        edge_attr = self.edge_tp(edge_attr, z, self.mlp(edge_len_emb))

        # Update the nodes
        e_sum = scatter(src=edge_attr, index=j, dim=0, dim_size=x.size(0))
        x = self.node_tp(x, e_sum)

        return x, edge_attr
