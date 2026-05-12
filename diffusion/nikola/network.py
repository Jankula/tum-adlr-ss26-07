import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# import PyTorch3D
# import diffusers
# from torch_geometric.nn.encoding import PositionalEncoding



class DiffusionSetup:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.t = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x_0, t):
        """Forward process: q(x_t | x_0)"""
        noise = torch.randn_like(x_0)
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t])[:, None, None]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod[t])[:, None, None]
        
        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
        return x_t, noise
    
class SineCosineEncoding(nn.Module):
    """Encodes the integer timestep 't' into a vector."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ShapeDiffusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SineCosineEncoding(128),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.model = nn.Sequential(
            nn.Linear(3 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3) # Predicts the 3D noise vector
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t) # [Batch, 128]
        t_emb = t_emb.unsqueeze(1).repeat(1, x.shape[1], 1) # [Batch, N, 128]
        x_input = torch.cat([x, t_emb], dim=-1) # [Batch, N, 131]
        return self.model(x_input)
    


@torch.no_grad()
def sample_ddpm(model, diffuser, n_points=2048):
    model.eval()
    # 1. Start with pure noise
    x = torch.randn(1, n_points, 3)
    
    # 2. Step backwards from T to 1
    for i in reversed(range(1000)):
        t = torch.tensor([i])
        predicted_noise = model(x, t)
        
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
def sample_ddim(model, diffuser, n_points=2048, steps=50):
    model.eval()
    x = torch.randn(1, n_points, 3)
    
    # Define the sparse schedule (e.g., [980, 960, ..., 0])
    times = torch.linspace(999, 0, steps).long()
    
    for i in range(len(times)):
        t = times[i].unsqueeze(0)
        prev_t = times[i+1].unsqueeze(0) if i+1 < len(times) else torch.tensor([-1])
        
        # 1. Predict noise
        pred_noise = model(x, t)
        
        # 2. Get alpha values for current and previous step
        alpha_t = diffuser.alpha_cumprod[t]
        alpha_prev = diffuser.alpha_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        
        # 3. Calculate "predicted x0" (the clean shape)
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        
        # 4. Calculate direction pointing to x_t
        direction_xt = torch.sqrt(1 - alpha_prev) * pred_noise
        
        # 5. Update x
        x = torch.sqrt(alpha_prev) * pred_x0 + direction_xt
        
    return x