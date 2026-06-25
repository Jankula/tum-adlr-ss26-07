import torch.nn as nn
import torch.nn.functional as F
import torch

from common import *


class Diffuser(nn.Module):
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.T = timesteps
        
        beta = torch.linspace(beta_start, beta_end, timesteps)
        alpha = 1.0 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alpha_cumprod))
        self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1.0 - alpha_cumprod))

    def add_noise(self, x_0, t):
        """Forward process: q(x_t | x_0)"""
        noise = torch.randn_like(x_0)
        
        # We use [t] to index, then [:, None, None] to match [Batch, N, 3]
        s_alpha = self.sqrt_alpha_cumprod[t][:, None, None]
        s_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t][:, None, None]
        
        x_t = s_alpha * x_0 + s_one_minus_alpha * noise
        return x_t, noise
    


class PointwiseNet(nn.Module):

    def __init__(self, latent_dim, hidden_dim, embedding_dim):
        super().__init__()
        assert(embedding_dim % 12 == 0)

        self.act = F.leaky_relu
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            ConcatSquashLinear(latent_dim, self.hidden_dim, 2*embedding_dim),
            ConcatSquashLinear(self.hidden_dim, 2 * self.hidden_dim, 2*embedding_dim),
            ConcatSquashLinear(2 * self.hidden_dim, 4 * self.hidden_dim, 2*embedding_dim),
            ConcatSquashLinear(4 * self.hidden_dim, 2 * self.hidden_dim, 2*embedding_dim),
            ConcatSquashLinear(2 * self.hidden_dim, self.hidden_dim, 2*embedding_dim),
            ConcatSquashLinear(self.hidden_dim, latent_dim, 2*embedding_dim)
        ])
        self.time_embedding = SineCosineEncoding(embedding_dim)
        self.grasp_embedding = SineCosineEncoding(embedding_dim // 12)
        self.grasp_project = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU()
        )

    
    def forward(self, x, time, grasp):
        """
        Args:
            x:  latent code at some timestep t, (B, F).
            time:     Time. (B, ).
            grasp:  Shape latents. (B, 12).
        """
        time_emb = self.time_embedding(time) # [B,F]
        
        grasp_emb = self.grasp_embedding(grasp) # [B,12,F/12]
        grasp_emb = grasp_emb.flatten(start_dim=1) # [B,F]
        grasp_emb = self.grasp_project(grasp_emb) # [B,F]

        ctx_emb = torch.cat([time_emb, grasp_emb], dim=-1) # [B,F+F]

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        return out

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=128, embedding_dim=120, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.diffuser = Diffuser(timesteps, beta_start, beta_end)
        self.denoiser = PointwiseNet(latent_dim, hidden_dim, embedding_dim)


@torch.no_grad()
def sample_ddpm(decoder, code, n_points=2048, timesteps=1000):
    """Generate a sample using the DDPM reverse diffusion process.

    Args:
        decoder: Decoder model containing the diffuser and denoiser modules.
        code: Latent code used as conditioning input for the denoiser, shape (B, F).
        n_points: Number of points to generate in the output point cloud.

    Returns:
        Generated point cloud tensor of shape (1, n_points, 3).
    """
    diffuser = decoder.diffuser
    denoiser = decoder.denoiser

    denoiser.eval()
    device = diffuser.beta.device
    # 1. Start with pure noise
    x = torch.randn(1, n_points, 3, device=device)
    
    # 2. Step backwards from T to 1
    for i in reversed(range(timesteps)):
        t = torch.tensor([i], device=device)
        beta = diffuser.beta[t]
        beta = beta.to(device)

        predicted_noise = denoiser(x, beta, code)
        
        # Math for the reverse step
        alpha = diffuser.alpha[i]
        alpha_cumprod = diffuser.alpha_cumprod[i]
        beta = diffuser.beta[i]
        
        noise_factor = (1 - alpha) / torch.sqrt(1 - alpha_cumprod)
        x = (1 / torch.sqrt(alpha)) * (x - noise_factor * predicted_noise)
        
        if i > 0: # Add a little bit of randomness back in (Langevin dynamics)
            x += torch.sqrt(beta) * torch.randn_like(x)
            
    return x


@torch.no_grad()
def sample_ddim(decoder, code, n_points=2048, steps=50, timesteps=1000):
    """Generate a sample using the DDIM reverse diffusion process.

    Args:
        decoder: Decoder model containing the diffuser and denoiser modules.
        code: Latent code used as conditioning input for the denoiser, shape (B, F).
        n_points: Number of points to generate in the output point cloud.
        steps: Number of inference steps to use in the DDIM scheduler.

    Returns:
        Generated point cloud tensor of shape (1, n_points, 3).
    """
    
    diffuser = decoder.diffuser
    denoiser = decoder.denoiser
    
    denoiser.eval()
    device  = diffuser.beta.device
    x = torch.randn(1, n_points, 3, device=device)
    
    # Define the sparse schedule (e.g., [980, 960, ..., 0])
    times = torch.linspace(timesteps-1, 0, steps).long()
    
    for i in range(len(times)):
        t = times[i].unsqueeze(0)
        t = t.to(device)
        prev_t = times[i+1].unsqueeze(0) if i+1 < len(times) else torch.tensor([-1])
        prev_t = prev_t.to(device)
        
        pred_noise = denoiser(x, t, code)
        
        alpha_t = diffuser.alpha_cumprod[t]
        alpha_prev = diffuser.alpha_cumprod[prev_t] if prev_t >= 0 else torch.tensor([1.0], device=device)
        
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        
        direction_xt = torch.sqrt(1 - alpha_prev) * pred_noise
        
        x = torch.sqrt(alpha_prev) * pred_x0 + direction_xt
        
    return x