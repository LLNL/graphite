import torch
from ..order import steinhardt

from e3nn.nn import FullyConnectedNet


class SteinhardtNet(torch.nn.Module):
    """Steinhardt featurization followed by MLP.

    Args:
        ls (list of ints): List of `l` (angular momentum numbers).
        head_neurons (list of ints): Number of neurons per layers in the MLP that transforms the Steinhardt features to final output.
            For hidden and last layers, not the first layer.
    """
    def __init__(self, ls = [4, 6], head_neurons = [64, 4]):
        super().__init__()

        self.ls   = ls
        self.head = FullyConnectedNet([len(ls)*2] + head_neurons, torch.nn.functional.silu)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        q = torch.zeros(len(x), len(self.ls), dtype=torch.float, device=x.device)
        w = torch.zeros(len(x), len(self.ls), dtype=torch.float, device=x.device)


        for idx, l in enumerate(self.ls):
            ql, wl = steinhardt(l, edge_index, edge_attr)
            q[:, idx] = ql
            w[:, idx] = torch.nan_to_num(wl)  # For some reason there tend to be a few nan's in `wl`.

        emb = torch.cat((q, w), dim=1)

        if self.training:
            return self.head(emb)
        else:
            return self.head(emb), emb
