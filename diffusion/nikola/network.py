from matplotlib.pylab import beta
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# import PyTorch3D
# import diffusers
# from torch_geometric.nn.encoding import PositionalEncoding



class Diffuser(nn.Module):
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.t = timesteps
        
        # 1. Calculate values locally
        beta = torch.linspace(beta_start, beta_end, timesteps)
        alpha = 1.0 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        
        # 2. Register buffers (This moves them to GPU automatically with .to(device))
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        
        # Pre-calculating these saves CPU/GPU cycles during training
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alpha_cumprod))
        self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1.0 - alpha_cumprod))

    def add_noise(self, x_0, t):
        """Forward process: q(x_t | x_0)"""
        # noise must be on the same device as x_0
        noise = torch.randn_like(x_0)
        
        # Access the buffers directly using self.name
        # We use [t] to index, then [:, None, None] to match [Batch, N, 3]
        s_alpha = self.sqrt_alpha_cumprod[t][:, None, None]
        s_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t][:, None, None]
        
        x_t = s_alpha * x_0 + s_one_minus_alpha * noise
        return x_t, noise
    
class SineCosineEncoding(nn.Module):
    """Encodes the integer timestep 't' into a vector."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t = t.float().view(-1, 1)
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        # emb = t * emb[None, :]
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Denoiser_dumb(nn.Module):
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
            nn.Linear(512, 512),
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

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            # LayerNorm is perfect for [Batch, Points, Channels] 
            # as it normalizes across the last dimension (dim).
            nn.LayerNorm(dim), 
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        # x shape: [B, N, dim]
        return x + self.block(x)

class Denoiser(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SineCosineEncoding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Initial projection of [x, y, z]
        self.input_proj = nn.Linear(3, hidden_dim)
        
        # Processing layers
        self.layer1 = nn.Linear(hidden_dim + hidden_dim, hidden_dim) # feat + time
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Deep residual layers with global context
        # Input to these will be (local_feat + global_feat + time)
        self.res_layers = nn.ModuleList([
            nn.Linear(hidden_dim * 3, hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])
        
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3) # Output: predicted noise [B, N, 3]
        )

    def forward(self, x, t):
        # x: [Batch, N, 3], t: [Batch]
        B, N, _ = x.shape
        
        # 1. Time Embedding
        t_emb = self.time_mlp(t) # [B, hidden_dim]
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, N, -1) # [B, N, hidden_dim]
        
        # 2. Local Features
        feat = self.input_proj(x) # [B, N, hidden_dim]
        
        # 3. Global Context (PointNet style)
        # We need [B, hidden_dim, N] for pooling
        global_feat = torch.max(feat, dim=1, keepdim=True)[0] # [B, 1, hidden_dim]
        global_feat_expanded = global_feat.expand(-1, N, -1)
        
        # 4. Combine and Refine
        # Concatenate Local + Global + Time
        combined = torch.cat([feat, global_feat_expanded, t_emb_expanded], dim=-1)
        
        # Pass through residual stack
        x_out = self.res_layers[0](combined)
        for i in range(1, len(self.res_layers)):
            x_out = self.res_layers[i](x_out)
            
        return self.final_proj(x_out)
    


@torch.no_grad()
def sample_ddpm(model, diffuser, n_points=2048):
    model.eval()
    device = diffuser.beta.device
    # 1. Start with pure noise
    x = torch.randn(1, n_points, 3, device=device)
    
    # 2. Step backwards from T to 1
    for i in reversed(range(1000)):
        t = torch.tensor([i], device=device)
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
    device  = diffuser.beta.device
    x = torch.randn(1, n_points, 3, device=device)
    
    # Define the sparse schedule (e.g., [980, 960, ..., 0])
    times = torch.linspace(999, 0, steps).long()
    
    for i in range(len(times)):
        t = times[i].unsqueeze(0)
        t = t.to(device)
        prev_t = times[i+1].unsqueeze(0) if i+1 < len(times) else torch.tensor([-1])
        prev_t = prev_t.to(device)
        
        # 1. Predict noise
        pred_noise = model(x, t)
        
        # 2. Get alpha values for current and previous step
        alpha_t = diffuser.alpha_cumprod[t]
        alpha_prev = diffuser.alpha_cumprod[prev_t] if prev_t >= 0 else torch.tensor([1.0], device=device)
        
        # 3. Calculate "predicted x0" (the clean shape)
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        
        # 4. Calculate direction pointing to x_t
        direction_xt = torch.sqrt(1 - alpha_prev) * pred_noise
        
        # 5. Update x
        x = torch.sqrt(alpha_prev) * pred_x0 + direction_xt
        
    return x

@torch.no_grad()
def sample_and_capture(denoiser, diffuser, n_points=2048, save_every=10):
    denoiser.eval()
    device = next(denoiser.parameters()).device
    
    # 1. Initialize starting noise
    x = torch.randn(1, n_points, 3).to(device)
    samples_list = []
    
    # 2. Reverse Loop
    for i in reversed(range(1000)):
        t = torch.tensor([i], device=device)
        predicted_noise = denoiser(x, t)
        
        # Get constants from diffuser
        alpha = diffuser.alpha[i]
        alpha_cumprod = diffuser.alpha_cumprod[i]
        beta = diffuser.beta[i]
        
        # Reverse Step Calculation
        noise_factor = (1 - alpha) / torch.sqrt(1 - alpha_cumprod)
        x = (1 / torch.sqrt(alpha)) * (x - noise_factor * predicted_noise)
        
        if i > 0:
            x += torch.sqrt(beta) * torch.randn_like(x)
        
        # 3. Save snapshot
        # We use .cpu() and .clone() to keep GPU memory free
        if i % save_every == 0 or i == 0:
            samples_list.append(x.detach().cpu().clone())
            
    return x, samples_list