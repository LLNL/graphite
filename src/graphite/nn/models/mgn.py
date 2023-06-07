import torch
from torch import nn
import copy

from ..mlp import MLP
from ..basis import GaussianRandomFourierFeatures
from ..conv import MeshGraphNetsConv


class Encoder(nn.Module):
    """MeshGraphNets Encoder.
    The encoder must take a PyG graph object `data` and output the same `data`
    with additional fields `h_node` and `h_edge` that correspond to the node and edge embedding.
    """
    def __init__(self, num_species, node_dim, edge_dim):
        super().__init__()
        self.num_species = num_species
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.embed_node = nn.Sequential(
            MLP([num_species, node_dim, node_dim], act=nn.SiLU()),
            nn.LayerNorm(node_dim),
        )
        self.embed_edge = nn.Sequential(
            MLP([3+1, edge_dim, edge_dim], act=nn.SiLU()),
            nn.LayerNorm(edge_dim),
        )
    
    def forward(self, data):
        # Embed nodes
        data.h_node = self.embed_node(data.z)

        # Embed edges
        e_ij = data.edge_attr
        h_edge = torch.cat([e_ij, e_ij.norm(dim=-1, keepdim=True)], dim=-1)
        data.h_edge = self.embed_edge(h_edge)
        
        return data


class Encoder_dpm(Encoder):
    """MeshGraphNets Encoder for diffusion model, with additional time encoding `data.h_time`.
    """
    def __init__(self, num_species, node_dim, edge_dim):
        super().__init__(num_species, node_dim, edge_dim)
        self.embed_time = nn.Sequential(
            GaussianRandomFourierFeatures(node_dim, input_dim=1),
            MLP([node_dim, node_dim, node_dim], act=nn.SiLU()),
            nn.LayerNorm(node_dim),
        )

    def forward(self, data, t):
        data = super().forward(data)
        data.h_node += self.embed_time(t)[data.batch].squeeze(1)
        return data


class Processor(nn.Module):
    """MeshGraphNets Processor.
    The processor updates both node and edge embeddings `data.h_node`, `data.h_edge`.
    """
    def __init__(self, num_convs, node_dim, edge_dim):
        super().__init__()
        self.num_convs = num_convs
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.convs = nn.ModuleList(
            [copy.deepcopy(MeshGraphNetsConv(node_dim, edge_dim)) for _ in range(num_convs)]
        )

    def forward(self, data):
        for conv in self.convs:
            data.h_node, data.h_edge = conv(data.h_node, data.edge_index, data.h_edge)
        return data


class Decoder(nn.Module):
    """MeshGraphNets Decoder.
    This decoder only operates on the node embedding `data.h_node`.
    """
    def __init__(self, node_dim, out_dim):
        super().__init__()
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.decoder = MLP([node_dim, node_dim, out_dim], act=nn.SiLU())
    
    def forward(self, data):
        return self.decoder(data.h_node)


class MeshGraphNets(nn.Module):
    """MeshGraphNets.
    """
    def __init__(self, encoder, processor, decoder):
        super().__init__()
        self.encoder   = encoder
        self.processor = processor
        self.decoder   = decoder
    
    def forward(self, data):
        data = self.encoder(data)
        data = self.processor(data)
        return self.decoder(data)


class MeshGraphNets_dpm(MeshGraphNets):
    """Time-dependent MeshGraphNets for diffusion model.
    """
    def forward(self, data, t, sigma=1.0):
        data = self.encoder(data, t)
        data = self.processor(data)
        return self.decoder(data) / sigma


class MeshGraphNets_dpm_cond(MeshGraphNets):
    """Time-dependent, conditional MeshGraphNets for conditional diffusion model.
    """
    def forward(self, x, x_cond, t):
        data = x_cond
        data = self.encoder(x, data, t)
        data = self.processor(data)
        return self.decoder(data)