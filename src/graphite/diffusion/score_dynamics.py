import torch
import torch.nn as nn


class ScoreDynamicsDiffuser:
    def __init__(self, sigma_scale=1.0, schedule='linear', t_min=1e-3):
        self.schedule = schedule
        self.t_min = t_min
        self.sigma_scale = sigma_scale

        if schedule == 'linear':
            beta_0, beta_1 = 0.1, 20
            self.t_max     = 1
            self.alpha_t   = lambda t: torch.exp(-(beta_0*t/2+t.pow(2)/4*(beta_1-beta_0)))
            self.t_lambda  = self.t_lambda = lambda lmd: 2*torch.log(torch.exp(-2*lmd)+1)/(torch.sqrt(\
                beta_0**2 + (beta_1-beta_0)*2*torch.log(torch.exp(-2*lmd)+1)) + beta_0)
        elif schedule == 'cosine':
            s             = 0.008
            self.t_max    = 0.9946
            f_sqrt        = lambda t: torch.cos(((t+s)/(1+s))*torch.pi/2)
            f0            = f_sqrt(torch.tensor(0)).item()
            self.alpha_t  = lambda t: f_sqrt(t)/f0
            self.t_lambda = lambda lmd: 2*(1+s)/torch.pi*torch.arccos((1/torch.sqrt(torch.exp(-2*lmd)+1))*f0)-s
        else:
            raise NotImplementedError(f'Unknown noise schedule {schedule}')

        self.sigma_t    = lambda t: torch.sqrt(1-self.alpha_t(t)**2)
        self.lambda_t   = lambda t: torch.log(self.alpha_t(t)/self.sigma_t(t))
        self.lambda_min = self.lambda_t(torch.tensor(self.t_min)).item()
        self.lambda_max = self.lambda_t(torch.tensor(self.t_max)).item()

    def forward_noise(self, x, t):
        eps = torch.randn_like(x)
        x_t = self.alpha_t(t)*x + self.sigma_t(t)*eps * self.sigma_scale
        return x_t, eps
    
    def reverse_denoise(self, x_T, noise_net, solver, M=10):
        """Apply reverse denoising to `x_T` (typically a random Gaussian noise)
        with a learned noise prediction model `noise_net`.
        """
        lambdas = torch.linspace(self.lambda_max, self.lambda_min, M+1, device=x_T.device).view(-1, 1, 1)
        t = self.t_lambda(lambdas)
        x = x_T
        xs = []
        for i in range(1, len(t)):
            x = solver(x, t[i-1], t[i], noise_net)
            xs.append(x.clone())
        return torch.stack(xs)

    def solver1(self, x, t_im1, t_i, noise_net):
        h_i = self.lambda_t(t_i) - self.lambda_t(t_im1)
        return self.alpha_t(t_i)/self.alpha_t(t_im1) * x - \
            self.sigma_t(t_i)*torch.expm1(h_i) * noise_net(x=x, t=t_im1)

    def solver2(self, x, t_im1, t_i, noise_net, r1=0.5):
        h_i  = self.lambda_t(t_i) - self.lambda_t(t_im1)
        s_i  = self.t_lambda(self.lambda_t(t_im1) + r1*h_i)
        T1   = noise_net(x=x, t=t_im1)
        t2   = self.sigma_t(t_i)*torch.expm1(h_i)
        u_i  = self.alpha_t(s_i)/self.alpha_t(t_im1) * x - self.sigma_t(s_i)*torch.expm1(r1*h_i) * T1
        # special case r1=0.5
        if r1 == 0.5:
            return self.alpha_t(t_i)/self.alpha_t(t_im1) * x - t2*noise_net(x=u_i, t=s_i)
        else:
            return self.alpha_t(t_i)/self.alpha_t(t_im1) * x - t2 * T1 - t2/(2*r1)*(noise_net(x=u_i, t=s_i) - T1)

    def solver3(self, x, t_im1, t_i, noise_net, r1=1/3, r2=2/3):
        h_i    = self.lambda_t(t_i) - self.lambda_t(t_im1)
        s_2im1 = self.t_lambda(self.lambda_t(t_im1) + r1*h_i)
        s_2i   = self.t_lambda(self.lambda_t(t_im1) + r2*h_i)
        T1     = noise_net(x=x, t=t_im1)
        u_2im1 = self.alpha_t(s_2im1)/self.alpha_t(t_im1) * x - self.sigma_t(s_2im1)*torch.expm1(r1*h_i) * T1
        D_2im1 = noise_net(x=u_2im1, t=s_2im1) - T1
        u_2i   = self.alpha_t(s_2i)/self.alpha_t(t_im1) * x - \
            self.sigma_t(s_2i)*torch.expm1(r2*h_i) * T1 - \
            self.sigma_t(s_2i)*r2/r1*(torch.expm1(r2*h_i)/(r2*h_i)-1) * D_2im1
        D_2i   = noise_net(x=u_2i, t=s_2i) - T1
        return self.alpha_t(t_i)/self.alpha_t(t_im1) * x - \
            self.sigma_t(t_i)*torch.expm1(h_i) * T1 - \
            self.sigma_t(t_i)/r2*(torch.expm1(h_i)/h_i-1) * D_2i
