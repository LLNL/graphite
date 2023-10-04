import torch
import torch.nn as nn
import numpy as np
from scipy import integrate


class VariancePreservingDiffuser:
    def __init__(self, t_min=1e-3, t_max=0.999):
        self.t_min = t_min
        self.t_max = t_max

        # Case 1: gamma = 1 - t, alpha = gamma.sqrt(), sigma = (1-gamma).sqrt()
        # self.gamma = lambda t: 1 - t
        # self.alpha = lambda t: self.gamma(t)**0.5
        # self.sigma = lambda t: (1 - self.gamma(t))**0.5
        # self.f     = lambda t: 0.5 / (t - 1)
        # self.g2    = lambda t: 1 - 2*self.f(t)*t
        # self.g     = lambda t: self.g2(t)**0.5

        # Case 2: alpha = cos(pi/2*t)
        self.alpha = lambda t: torch.cos(torch.pi/2*t)
        self.sigma = lambda t: torch.sin(torch.pi/2*t)
        self.f     = lambda t: torch.tan(torch.pi/2*t) * torch.pi * (-0.5)
        self.g2    = lambda t: torch.pi*self.alpha(t)*self.sigma(t) - 2*self.f(t)*(self.sigma(t)**2)
        self.g     = lambda t: self.g2(t)**0.5

    def forward_noise(self, x, t, b=1.0):
        alpha = self.alpha(t)
        sigma = self.sigma(t)
        eps = torch.randn_like(x)
        return alpha*x*b + sigma*eps, eps


class VarianceExplodingDiffuser:
    def __init__(self, k=1.0, t_min=1e-3, t_max=0.999):
        self.t_min = t_min
        self.t_max = t_max

        # Case 1: sigma = kt
        self.alpha = lambda t: 1
        self.sigma = lambda t: k*t
        self.f     = lambda t: 0
        self.g2    = lambda t: 2*(k**2)*t
        self.g     = lambda t: self.g2(t)**0.5

    def forward_noise(self, x, t):
        alpha = self.alpha(t)
        sigma = self.sigma(t)
        eps = torch.randn_like(x)
        return alpha*x + sigma*eps, eps
