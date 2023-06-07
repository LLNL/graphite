import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


class GatedGCN(MessagePassing):
    """Gated GCN, also known as edge-gated convolution.
    Reference: https://arxiv.org/abs/2003.00982
    
    Different from the original version, in this version, the activation function is SiLU,
    and the normalization is LayerNorm.

    This implementation concatenates the `x_i`, `x_j`, and `e_ij` feature vectors during the edge update.
    """
    def __init__(self, node_dim, edge_dim, epsilon=1e-5):
        super().__init__(aggr='add')
        self.W_src  = nn.Linear(node_dim, node_dim)
        self.W_dst  = nn.Linear(node_dim, node_dim)
        self.W_e    = nn.Linear(node_dim*2 + edge_dim, edge_dim)
        self.act    = nn.SiLU()
        self.sigma  = nn.Sigmoid()
        self.norm_x = nn.LayerNorm([node_dim])
        self.norm_e = nn.LayerNorm([edge_dim])
        self.eps    = epsilon

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W_src.weight); self.W_src.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_dst.weight); self.W_dst.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_e.weight);   self.W_e.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        i, j = edge_index

        # Calculate gated edges
        sigma_e = self.sigma(edge_attr)
        e_sum   = scatter(src=sigma_e, index=i, dim=0)
        e_gated = sigma_e / (e_sum[i] + self.eps)

        # Update the nodes (this utilizes the gated edges)
        out = self.propagate(edge_index, x=x, e_gated=e_gated)
        out = self.W_src(x) + out
        out = x + self.act(self.norm_x(out))

        # Update the edges
        z = torch.cat([x[i], x[j], edge_attr], dim=-1)
        edge_attr = edge_attr + self.act(self.norm_e(self.W_e(z)))

        return out, edge_attr

    def message(self, x_j, e_gated):
        return e_gated * self.W_dst(x_j)


class GatedGCN_v2(MessagePassing):
    """Gated GCN, also known as edge-gated convolution.
    Reference: https://arxiv.org/abs/2003.00982

    Different from the original version, in this version, the activation function is SiLU,
    and the normalization is LayerNorm.

    This implementation is closer to the original formulation (without the concatenation).
    """
    def __init__(self, node_dim, edge_dim, epsilon=1e-5):
        super().__init__(aggr='add')
        self.W_src  = nn.Linear(node_dim, node_dim)
        self.W_dst  = nn.Linear(node_dim, node_dim)
        self.W_A    = nn.Linear(node_dim, edge_dim)
        self.W_B    = nn.Linear(node_dim, edge_dim)
        self.W_C    = nn.Linear(edge_dim, edge_dim)
        self.act    = nn.SiLU()
        self.sigma  = nn.Sigmoid()
        self.norm_x = nn.LayerNorm([node_dim])
        self.norm_e = nn.LayerNorm([edge_dim])
        self.eps    = epsilon

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W_src.weight); self.W_src.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_dst.weight); self.W_dst.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_A.weight);   self.W_A.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_B.weight);   self.W_B.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_C.weight);   self.W_C.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        i, j = edge_index

        # Calculate gated edges
        sigma_e = self.sigma(edge_attr)
        e_sum   = scatter(src=sigma_e, index=i , dim=0)
        e_gated = sigma_e / (e_sum[i] + self.eps)

        # Update the nodes (this utilizes the gated edges)
        out = self.propagate(edge_index, x=x, e_gated=e_gated)
        out = self.W_src(x) + out
        out = x + self.act(self.norm_x(out))

        # Update the edges
        edge_attr = edge_attr + self.act(self.norm_e(self.W_A(x[i]) \
                                                   + self.W_B(x[j]) \
                                                   + self.W_C(edge_attr)))

        return out, edge_attr

    def message(self, x_j, e_gated):
        return e_gated * self.W_dst(x_j)
