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
        s_alpha = self.sqrt_alpha_cumprod[t][:, None]
        s_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t][:, None]
        
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


# @torch.no_grad()
# def sample_ddpm(decoder, grasp, n_points=2048, timesteps=1000):
#     """Generate a sample using the DDPM reverse diffusion process.

#     Args:
#         decoder: Decoder model containing the diffuser and denoiser modules.
#         code: Latent code used as conditioning input for the denoiser, shape (B, F).
#         n_points: Number of points to generate in the output point cloud.

#     Returns:
#         Generated point cloud tensor of shape (1, n_points, 3).
#     """
#     diffuser = decoder.diffuser
#     denoiser = decoder.denoiser

#     denoiser.eval()
#     device = diffuser.beta.device
#     # 1. Start with pure noise
#     x = torch.randn(1, n_points, 3, device=device)
    
#     # 2. Step backwards from T to 1
#     for i in reversed(range(timesteps)):
#         t = torch.tensor([i], device=device)
#         beta = diffuser.beta[t]
#         beta = beta.to(device)

#         predicted_noise = denoiser(x, beta, code)
        
#         # Math for the reverse step
#         alpha = diffuser.alpha[i]
#         alpha_cumprod = diffuser.alpha_cumprod[i]
#         beta = diffuser.beta[i]
        
#         noise_factor = (1 - alpha) / torch.sqrt(1 - alpha_cumprod)
#         x = (1 / torch.sqrt(alpha)) * (x - noise_factor * predicted_noise)
        
#         if i > 0: # Add a little bit of randomness back in (Langevin dynamics)
#             x += torch.sqrt(beta) * torch.randn_like(x)
            
#     return x

@torch.no_grad()
def sample_ddpm(decoder, grasp, latent_dim=128, timesteps=1000):
    """Generate a 128-dimensional latent code batch using the DDPM reverse diffusion process.

    Args:
        decoder: Decoder model containing the diffuser and denoiser modules.
        grasp: Grasp shape latents used as conditioning input, shape (B, 12).
        latent_dim: The dimensionality of your flat latent code (128).
        timesteps: Total number of diffusion timesteps (1000).

    Returns:
        Generated latent code tensor of shape (B, latent_dim).
    """
    diffuser = decoder.diffuser
    denoiser = decoder.denoiser
    denoiser.eval()

    device = diffuser.beta.device
    batch_size = grasp.shape[0]  # Dynamically match your input batch size

    # 1. Start with pure noise matching your [B, 128] latent vector dimension
    x = torch.randn(batch_size, latent_dim, device=device)
    
    # 2. Step backwards from T-1 down to 0
    for i in reversed(range(timesteps)):
        # Create a batch-sized time tensor for the denoiser
        t = torch.tensor([i], device=device).repeat(batch_size)
        
        # Predict the noise using the correct signature order: (x, time, grasp)
        predicted_noise = denoiser(x, t, grasp)
        
        # Grab scalar values for step calculations
        alpha = diffuser.alpha[i]
        alpha_cumprod = diffuser.alpha_cumprod[i]
        beta = diffuser.beta[i]
        
        # DDPM Reverse Step Math
        noise_factor = (1 - alpha) / torch.sqrt(1 - alpha_cumprod)
        x = (1 / torch.sqrt(alpha)) * (x - noise_factor * predicted_noise)
        
        # Add Langevin noise back in if we aren't at the very last step (i > 0)
        if i > 0:
            x += torch.sqrt(beta) * torch.randn_like(x)
            
    return x


# @torch.no_grad()
# def sample_ddim(decoder, grasp, latent_dim=128, steps=50, timesteps=1000):
#     """Generate a sample using the DDIM reverse diffusion process.

#     Args:
#         decoder: Decoder model containing the diffuser and denoiser modules.
#         steps: Number of inference steps to use in the DDIM scheduler.

#     Returns:
#         Generated point cloud tensor of shape (1, n_points, 3).
#     """
    
#     diffuser = decoder.diffuser
#     denoiser = decoder.denoiser
#     denoiser.eval()
    
#     device  = diffuser.beta.device
#     x = torch.randn(1, latent_dim, device=device)
    
#     # Define the sparse schedule (e.g., [980, 960, ..., 0])
#     times = torch.linspace(timesteps-1, 0, steps).long()
    
#     for i in range(len(times)):
#         t = times[i].unsqueeze(0)
#         t = t.to(device)
#         prev_t = times[i+1].unsqueeze(0) if i+1 < len(times) else torch.tensor([-1])
#         prev_t = prev_t.to(device)
        
#         pred_noise = denoiser(x, t, grasp)
        
#         alpha_t = diffuser.alpha_cumprod[t]
#         alpha_prev = diffuser.alpha_cumprod[prev_t] if prev_t >= 0 else torch.tensor([1.0], device=device)
        
#         pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        
#         direction_xt = torch.sqrt(1 - alpha_prev) * pred_noise
        
#         x = torch.sqrt(alpha_prev) * pred_x0 + direction_xt
        
#     return x

@torch.no_grad()
def sample_ddim(decoder, grasp, latent_dim=128, steps=50, timesteps=1000):
    """Generate a sample using the DDIM reverse diffusion process."""
    diffuser = decoder.diffuser
    denoiser = decoder.denoiser
    denoiser.eval()
    
    device = diffuser.beta.device
    batch_size = grasp.shape[0]  # Dynamically adapt to input batch size
    
    # Initialize noise matching your exact batch and latent dimensions
    x = torch.randn(batch_size, latent_dim, device=device)
    
    # Create steps + 1 points so we can cleanly pair up every step t with its prev_t
    # Example for 5 steps: [999, 749, 499, 249, 0, -1]
    times = torch.linspace(timesteps - 1, 0, steps).long().tolist()
    times.append(-1)  # Explicitly add the final boundary step
    
    for i in range(steps):
        # Extract scalar timesteps
        t_val = times[i]
        prev_t_val = times[i+1]
        
        # Create batch-sized time tensors
        t = torch.tensor([t_val], device=device).repeat(batch_size)
        
        # Predict the noise using the correct order: (x, time, grasp)
        pred_noise = denoiser(x, t, grasp)
        
        # Pull alpha parameters and shape them as [B, 1] for smooth broadcasting
        alpha_t = diffuser.alpha_cumprod[t].unsqueeze(-1)
        
        if prev_t_val >= 0:
            prev_t = torch.tensor([prev_t_val], device=device).repeat(batch_size)
            alpha_prev = diffuser.alpha_cumprod[prev_t].unsqueeze(-1)
        else:
            # When dropping below 0, alpha_prev is exactly 1.0 (no noise left)
            alpha_prev = torch.ones(batch_size, 1, device=device)
        
        # DDIM core update equations
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        direction_xt = torch.sqrt(1 - alpha_prev) * pred_noise
        
        x = torch.sqrt(alpha_prev) * pred_x0 + direction_xt
        
    return x