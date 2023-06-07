import torch
from torch import nn
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer

from e3nn.nn import BatchNorm
from ..segnn.o3_building_blocks import O3TensorProduct, O3TensorProductSwishGate
from ..segnn.instance_norm import InstanceNorm


class EdgeProcessor(nn.Module):
    """Equivairant edge processor.
    Args:
        node_irreps:
        edge_irreps:
    """
    def __init__(self, node_irreps, edge_irreps):
        super().__init__()
        irreps_in = (2*node_irreps + '1x0e').simplify()
        self.tp1 = O3TensorProductSwishGate(irreps_in1=irreps_in, irreps_in2=edge_irreps, irreps_out=edge_irreps)
        # self.tp2 = O3TensorProduct(irreps_in1=edge_irreps, irreps_in2=edge_irreps, irreps_out=edge_irreps)
        self.norm = BatchNorm(edge_irreps)

    def forward(self, x_i, x_j, edge_attr, edge_len):
        out  = torch.cat([x_i, x_j, edge_len], dim=-1)
        out  = self.tp1(out, edge_attr)
        # out  = self.tp2(out, edge_attr)
        out  = self.norm(out)
        out += edge_attr
        return out


class NodeProcessor(nn.Module):
    """Equivariant node processor.
    Args:
        node_irreps:
        edge_irreps:
    """
    def __init__(self, node_irreps, edge_irreps):
        super().__init__()
        irreps_in = (node_irreps + edge_irreps).simplify()
        self.tp1 = O3TensorProductSwishGate(irreps_in1=node_irreps, irreps_in2=edge_irreps, irreps_out=node_irreps)
        # self.tp2 = O3TensorProduct(irreps_in1=node_irreps, irreps_in2=edge_irreps, irreps_out=node_irreps)
        self.norm = BatchNorm(node_irreps)

    def forward(self, x, edge_index, edge_attr):
        i, j = edge_index
        out = self.tp1(x[i], edge_attr)
        out = scatter(out, index=i, dim=0)
        # out  = self.tp1(x,   edge_attr_summed)
        # out  = self.tp2(out, edge_attr_summed)
        out  = self.norm(out)
        out += x
        return out


class EquivariantMGNConv(nn.Module):
    def __init__(self, node_irreps, edge_irreps):
        super().__init__()
        self.edge_model = EdgeProcessor(node_irreps, edge_irreps)
        self.node_model = NodeProcessor(node_irreps, edge_irreps)

    def forward(self, x, edge_index, edge_attr, edge_len):
        i, j = edge_index
        edge_attr = self.edge_model(x[i], x[j], edge_attr, edge_len)
        node_attr = self.node_model(x, edge_index, edge_attr)
        return node_attr, edge_attr