import torch
from torch import nn
from ..conv    import GatedGCN


class GatedGCN_Net(torch.nn.Module):
    """GatedGCN model from https://arxiv.org/abs/2003.00982.
    """
    def __init__(self, init_embed, output_layer, dim=100, num_convs=3, out_dim=1):
        super().__init__()
        self.init_embed = init_embed
        
        self.convs = nn.ModuleList([GatedGCN(dim, dim) for _ in range(num_convs)])

        self.out = output_layer
                
    def forward(self, data):
        data = self.init_embed(data)
        edge_index, h_node, h_edge = data.edge_index, data.h_node, data.h_edge

        for conv in self.convs:
            h_node, h_edge = conv(h_node, edge_index, h_edge)

        return self.out(h_node)