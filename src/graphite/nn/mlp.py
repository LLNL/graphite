import torch
from torch import nn


__all__ = ['MLP']


class MLP(nn.Module):
    """Multi-layer perceptron with custom activation functions.

    Args:
        hs (list of int): Input, hidden, and output dimensions.
        act (torch activation function, or None): Activation function that
            applies to all but the output layer. For example, `torch.nn.ReLU()`.
            If None, no activation function is applied.
    """
    def __init__(self, hs, act=None):
        super().__init__()
        
        num_layers = len(hs)

        layers = []
        for i in range(num_layers-1):
            layers += [nn.Linear(hs[i], hs[i+1])]
            if (act is not None) and (i < num_layers-2):
                layers += [act]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
