import torch
from torch import nn
from torch_geometric.utils import scatter

from e3nn      import o3
from e3nn.nn   import FullyConnectedNet

# Typing
from torch import Tensor
from typing import List, Optional, Tuple


class Interaction(nn.Module):
    """NequIP equivariant interaction/convolution layer.
    
    References:
    - https://arxiv.org/pdf/2101.03164.pdf
    - https://docs.e3nn.org/en/stable/api/nn/models/gate_points_2101.html
    - https://github.com/e3nn/e3nn/tree/0.4.4/e3nn/nn/models/v2106.
    - https://github.com/mir-group/nequip/blob/main/nequip/nn/_interaction_block.py
    """
    def __init__(self,
        irreps_in, irreps_node, irreps_edge, irreps_out, radial_neurons=[16, 64], num_neighbors=1):
        """
        Args:
            irreps_in (Irreps or str): Irreps of input node features.
            irreps_node (Irreps or str): Irreps of node attributes (constant throughout model).
            irreps_edge (Irreps or str): Irreps of edge spherical harmonics (constant througout model).
            irreps_out (Irreps or str): Irreps of output node features.
            radial_neurons (list of ints): Number of neurons per layers in the radial MLP.
                For first and hidden layers, not the output layer.
            num_neighbors (float): Typical or averaged node degree (used for normalization).
        """
        super().__init__()

        self.irreps_in     = o3.Irreps(irreps_in)
        self.irreps_node   = o3.Irreps(irreps_node)
        self.irreps_edge   = o3.Irreps(irreps_edge)
        self.irreps_out    = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        assert irreps_mid.dim > 0, (
            f"irreps_in={self.irreps_in} times irreps_edge={self.irreps_edge} "
            f"produces nothing in irreps_out={self.irreps_out}."
        )

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        self.sc   = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_node, self.irreps_out)
        self.lin1 = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_node, self.irreps_in)
        # self.lin1 = o3.Linear(self.irreps_in, self.irreps_in)
        self.conv = o3.TensorProduct(
            self.irreps_in,
            self.irreps_edge,
            irreps_mid,
            instructions,
            internal_weights = False,
            shared_weights   = False,
        )
        self.lin2 = o3.FullyConnectedTensorProduct(irreps_mid, self.irreps_node, self.irreps_out)
        # self.lin2 = o3.Linear(irreps_mid.simplify(), self.irreps_out)

        self.mlp = FullyConnectedNet(radial_neurons + [self.conv.weight_numel], torch.nn.functional.silu)

        # SkipInit mechanism inspired by https://arxiv.org/pdf/2002.10444.pdf
        self.alpha = o3.FullyConnectedTensorProduct(irreps_mid, self.irreps_node, "0e")
        with torch.no_grad():
            self.alpha.weight.zero_()
        assert self.alpha.output_mask[0] == 1.0, (
            f"irreps_mid={irreps_mid} and irreps_node={self.irreps_node} are not able to generate scalars."
        )

    def forward(self, x, node_attr, edge_index, edge_attr, edge_len_emb):
        i, j = edge_index
        num_nodes = x.size(0)

        node_self_connection = self.sc(x, node_attr)

        node_features = self.lin1(x, node_attr)
        # node_features = self.lin1(x)
        edge_features = self.conv(node_features[i], edge_attr, weight=self.mlp(edge_len_emb))
        node_features = scatter(edge_features, j, dim=0, dim_size=num_nodes).div(self.num_neighbors**0.5)
        node_conv_out = self.lin2(node_features, node_attr)
        # node_conv_out = self.lin2(node_features)

        alpha = self.alpha(node_features, node_attr)
        m = self.sc.output_mask
        alpha = (1 - m) + alpha * m
        return node_self_connection + alpha * node_conv_out
        # return node_self_connection + node_conv_out
