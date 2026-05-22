import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np

from DPM_common import *
from vae_model import *
from train_vae import KLD_loss

class VarianceSchedule(Module):

    def __init__(self, num_steps, beta_1=1e-4, beta_T=0.05, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility=0):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas
    
class PointwiseNet(Module):

    def __init__(self, point_dim, context_dim, hidden_dim, residual=True):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.hidden_dim = hidden_dim
        self.layers = ModuleList([
            ConcatSquashLinear(3, self.hidden_dim, context_dim+point_dim),
            ConcatSquashLinear(self.hidden_dim, 2 * self.hidden_dim, context_dim+point_dim),
            ConcatSquashLinear(2 * self.hidden_dim, 4 * self.hidden_dim, context_dim+point_dim),
            ConcatSquashLinear(self.hidden_dim * 4, 2 * self.hidden_dim, context_dim+point_dim),
            ConcatSquashLinear(2 * self.hidden_dim, self.hidden_dim, context_dim+point_dim),
            ConcatSquashLinear(self.hidden_dim, point_dim, context_dim+point_dim)
        ])

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out
        
class DiffusionPoint(Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, context, t=None):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            context:  Shape latent, (B, F).
        """
        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)
        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)

        loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        return loss

    def sample(self, num_points, context, point_dim=3, flexibility=0.0, ret_traj=False):
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]
            e_theta = self.net(x_t, beta=beta, context=context)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
        
        if ret_traj:
            return traj
        else:
            return traj[0]
        
        
class AutoEncoder(Module):

    def __init__(self, number_points=256, point_dim=3, hidden_dim=32, latent_dim=32, num_steps=1000, beta_1=1e-4, beta_T=0.05):
        super().__init__()
        self.encoder = PointNetEncoder(number_points, point_dim, 2 * hidden_dim, latent_dim)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=point_dim, context_dim=latent_dim, hidden_dim=hidden_dim, residual=True),
            var_sched = VarianceSchedule(
                num_steps=num_steps,
                beta_1=beta_1,
                beta_T=beta_T,
                mode="linear"
            )
        )

    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        mean, log_variance = self.encoder(x)
        
        return self.sample_latent_z(mean, log_variance), mean, log_variance
    
    def sample_latent_z(self, mean, log_variance):
        std = torch.exp(0.5 * log_variance) # compute std from log_variance
        eps = torch.randn_like(mean)
        return mean + eps * std

    def decode(self, code, num_points, flexibility=0.0, ret_traj=False):
        return self.diffusion.sample(num_points, code, flexibility=flexibility, ret_traj=ret_traj)

    def get_loss(self, x):
        code, mean, log_variance = self.encode(x)
        encoder_loss = 0.01 * KLD_loss(mean, log_variance)
        denoiser_loss = self.diffusion.get_loss(x, code)
        return denoiser_loss + encoder_loss, encoder_loss