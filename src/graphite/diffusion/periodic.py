import torch
import torch.nn as nn
import numpy as np
from scipy import integrate
from torch_geometric.data import Data


class PeriodicStructureDiffuser:
    """Diffuser implementation for periodic structures.

    References:
    - https://arxiv.org/pdf/2011.13456.pdf
    - https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=PpJSwfyY6mJz
    """
    def __init__(self, k=1.0, t_min=1e-3, t_max=0.999):
        self.t_min = t_min
        self.t_max = t_max
        self.k = k  # scaling factor for noise added to r

        self.gamma = lambda t: 1 - t

        # Noise schedule for atom types (z)
        self.alpha_z = lambda t: self.gamma(t)**0.5
        self.sigma_z = lambda t: (1 - self.gamma(t))**0.5
        self.f_z     = lambda t: 0.5 / (t - 1)
        self.g2_z    = lambda t: 1 - 2*self.f_z(t)*t
        self.g_z     = lambda t: self.g2_z(t)**0.5

        # Noise schedule for atomic positions (r)
        # Case 1: sigma = kt
        self.alpha_r = lambda t: 1
        self.sigma_r = lambda t: k*t
        self.f_r     = lambda t: 0
        self.g2_r    = lambda t: 2*(k**2)*t
        self.g_r     = lambda t: self.g2_r(t)**0.5
        # Case 2: sigma^2 = kt
        # self.alpha_r = lambda t: 1
        # self.sigma_r = lambda t: (k*t)**0.5
        # self.f_r     = lambda t: 0
        # self.g2_r    = lambda t: k
        # self.g_r     = lambda t: k**0.5
    
    def forward_noise_r(self, pos, t):
        sigma = self.sigma_r(t)
        eps   = torch.randn_like(pos)
        return pos + sigma*eps, sigma, eps

    def forward_noise_z(self, z, t):
        alpha = self.alpha_z(t)
        sigma = self.sigma_z(t)
        eps   = torch.randn_like(z)
        return alpha*z + sigma*eps, sigma, eps

    # def forward_noise_rz(self, data, t):
    #     data.t = t
    #     data.eps_z = torch.randn_like(data.z)
    #     data.eps_r = torch.randn_like(data.pos)
    #     data.alpha_z = self.alpha_z(t)
    #     data.sigma_z = self.sigma_z(t)
    #     data.sigma_r = self.sigma_r(t)
    #     if data.batch is not None:
    #         data.alpha_z = data.alpha_z[data.batch]
    #         data.sigma_z = data.sigma_z[data.batch]
    #         data.sigma_r = data.sigma_r[data.batch]
    #     data.z   = data.alpha_z*data.z + data.sigma_z*data.eps_z
    #     data.pos = data.pos            + data.sigma_r*data.eps_r
    #     return data

    def reverse_denoise_r_by_sde(self, z, pos, cell, score_fn, ts=torch.linspace(0.999, 0.001, 101)):
        ts = ts.to(pos.device).view(-1, 1)
        # dt = ts[1] - ts[0]
        pos_traj = [pos.clone()]
        for i, t in enumerate(ts[1:]):
            dt = ts[i+1] - ts[i]
            eps = dt.abs().sqrt() * torch.randn_like(pos)
            score = score_fn(z, pos, cell, t, sigma=self.sigma_r(t))
            disp = (self.f_r(t)*pos - self.g2_r(t)*score)*dt + self.g_r(t)*eps
            pos += disp
            pos_traj.append(pos.clone())
        return torch.stack(pos_traj)

    def reverse_denoise_r_by_ode(self, z, pos, cell, score_fn, rtol=1e-3, atol=1e-6, t_eval=None):
        x_shape = pos.shape
        device  = pos.device

        def ode_func(t, x):
            x = torch.tensor(x, device=device, dtype=torch.float).reshape(x_shape)
            t = torch.tensor(t, device=device, dtype=torch.float).view(-1)
            score = score_fn(z, pos, cell, t, sigma=self.sigma_r(t))
            out = - 0.5*self.g2_r(t)*score
            return out.cpu().numpy().reshape(-1).astype(np.float64)

        t_span = (self.t_max, self.t_min)
        x_init = pos.reshape(-1).cpu().numpy()
        sol = integrate.solve_ivp(ode_func, t_span, x_init, rtol=rtol, atol=atol, t_eval=t_eval)
        return sol

    # def noise_loss_rz(self, noise_net, data):
    #     pred_eps = noise_net(data, data.t)
    #     return nn.functional.mse_loss(pred_eps, torch.hstack([data.eps_r, data.eps_z]))

    # def score_loss_rz(self, score_net, data):
    #     score = score_net(data, data.t)
    #     score_sigma = torch.hstack([score[:,:3]*data.sigma_r, score[:, 3:]*data.sigma_z])
    #     eps = torch.hstack([data.eps_r, data.eps_z])
    #     return (score_sigma + eps).pow(2).sum(dim=-1).mean()

    # def reverse_denoise_rz(self, x_T, noise_net, rtol=1e-3, atol=1e-6, t_eval=None):
    #     x_shape = x_T.shape
    #     device = x_T.device

    #     def ode_func(t, x):
    #         x = torch.tensor(x, device=device, dtype=torch.float).reshape(x_shape)
    #         t = torch.tensor(t, device=device, dtype=torch.float).view(-1)
    #         z = x[:, 3:]
    #         with torch.no_grad():
    #             pred_eps = noise_net(x=x, t=t)
    #             pred_eps_r = pred_eps[:, :3]
    #             pred_eps_z = pred_eps[:, 3:]
    #             out_r =                 0.5*self.g2_r(t)*pred_eps_r/self.sigma_r(t)
    #             out_z = self.f_z(t)*z + 0.5*self.g2_z(t)*pred_eps_z/self.sigma_z(t)
    #             out   = torch.hstack([out_r, out_z])
    #         return out.cpu().numpy().reshape(-1).astype(np.float64)

    #     t_span = (self.t_max, self.t_min)
    #     x_init = x_T.reshape(-1).cpu().numpy()
    #     sol = integrate.solve_ivp(ode_func, t_span, x_init, rtol=rtol, atol=atol, t_eval=t_eval)
    #     return sol
