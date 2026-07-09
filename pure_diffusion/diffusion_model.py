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

    def __init__(self, hidden_dim, embedding_dim):
        super().__init__()
        assert(embedding_dim % 12 == 0)

        self.act = F.leaky_relu
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            ConcatSquashLinear(3, self.hidden_dim, 2*embedding_dim),
            ConcatSquashLinear(self.hidden_dim, 2 * self.hidden_dim, 2*embedding_dim),
            ConcatSquashLinear(2 * self.hidden_dim, 4 * self.hidden_dim, 2*embedding_dim),
            ConcatSquashLinear(4 * self.hidden_dim, 2 * self.hidden_dim, 2*embedding_dim),
            ConcatSquashLinear(2 * self.hidden_dim, self.hidden_dim, 2*embedding_dim),
            ConcatSquashLinear(self.hidden_dim, 3, 2*embedding_dim)
        ])
        self.time_embedding = SineCosineEncoding(embedding_dim)
        self.grasp_embedding = SineCosineEncoding(embedding_dim // 12)
        self.grasp_project = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU()
        )

    
    def forward(self, pc, grasp, time):
        """
        Args:
            pc:  Point clouds at some timestep t, (B, N, 3).
            grasp:  Shape latents. (B, 12).
            time:     Time. (B, ).
        """
        time_emb = self.time_embedding(time) # [B,F]
        
        grasp_emb = self.grasp_embedding(grasp) # [B,12,F/12]
        grasp_emb = grasp_emb.flatten(start_dim=1) # [B,F]
        grasp_emb = self.grasp_project(grasp_emb) # [B,F]

        ctx_emb = torch.cat([time_emb, grasp_emb], dim=-1) # [B,F+F]
        ctx_emb = ctx_emb.unsqueeze(1) # [B,1,F+F]


        out = pc
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        return out

class Decoder(nn.Module):
    def __init__(self, hidden_dim=128, embedding_dim=120, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.diffuser = Diffuser(timesteps, beta_start, beta_end)
        self.denoiser = PointwiseNet(hidden_dim, embedding_dim)


@torch.no_grad()
def sample_ddim(decoder, grasp, n_points=2048, steps=50, timesteps=1000):
    """Generate point clouds from grasp configurations using the DDIM reverse process.

    Args:
        decoder: Decoder model containing the diffuser and denoiser modules.
        grasp: Grasp configurations used as conditioning input, shape (B, 12).
        n_points: Number of points to generate per point cloud.
        steps: Number of accelerated inference steps (e.g., 50 instead of 1000).
        timesteps: Total number of diffusion training timesteps (e.g., 1000).

    Returns:
        Generated point cloud tensor of shape (B, n_points, 3).
    """
    diffuser = decoder.diffuser
    denoiser = decoder.denoiser
    denoiser.eval()
    
    device = diffuser.beta.device
    batch_size = grasp.shape[0]  # Dynamically capture batch size from grasp inputs

    # 1. Initialize point clouds with pure Gaussian noise: (B, N, 3)
    x = torch.randn(batch_size, n_points, 3, device=device)
    
    # 2. Define the sparse time trajectory from T-1 down to 0
    # (e.g., [999, 979, 959, ..., 0] if steps=50)
    times = torch.linspace(timesteps - 1, 0, steps, dtype=torch.long, device=device)
    
    for i in range(len(times)):
        # Current and next sparse timesteps
        t_val = times[i]
        prev_t_val = times[i + 1] if i + 1 < len(times) else torch.tensor(-1, device=device)
        
        # Broadcast time across the batch to match your denoiser's expected shape: (B,)
        t = t_val.expand(batch_size)
        
        # 3. Predict noise using your PointwiseNet forward signature: (pc, grasp, time)
        pred_noise = denoiser(pc=x, grasp=grasp, time=t)
        
        # 4. Fetch alpha_cumprod values for the step equations
        alpha_t = diffuser.alpha_cumprod[t_val]
        if prev_t_val >= 0:
            alpha_prev = diffuser.alpha_cumprod[prev_t_val]
        else:
            alpha_prev = torch.tensor(1.0, device=device) # Base case boundary condition at t=0
        
        # 5. DDIM Formulation Math (Deterministic, eta = 0)
        # Formula: x_{t-1} = sqrt(alpha_prev) * pred_x0 + direction_to_xt
        
        # Estimate the fully denoised clean point cloud (x_0)
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        
        # Direction pointing back to the noisy trajectory point
        direction_xt = torch.sqrt(1 - alpha_prev) * pred_noise
        
        # Step variation update
        x = torch.sqrt(alpha_prev) * pred_x0 + direction_xt
        
    return x

class PointwiseNet_mod(nn.Module):

    def __init__(self, hidden_dim, embedding_dim):
        super().__init__()
        self.act = F.leaky_relu
        self.hidden_dim = hidden_dim
        
        self.layers = nn.ModuleList([
            ConcatSquashLinear(3, self.hidden_dim, 2*embedding_dim),
            ConcatSquashLinear(self.hidden_dim, 2 * self.hidden_dim, 2*embedding_dim),
            ConcatSquashLinear(2 * self.hidden_dim, 4 * self.hidden_dim, 2*embedding_dim),
            ConcatSquashLinear(4 * self.hidden_dim, 2 * self.hidden_dim, 2*embedding_dim),
            ConcatSquashLinear(2 * self.hidden_dim, self.hidden_dim, 2*embedding_dim),
            ConcatSquashLinear(self.hidden_dim, 3, 2*embedding_dim)
        ])
        
        # Time gets Sin/Cos encoding
        self.time_embedding = SineCosineEncoding(embedding_dim)
        
        # Grasp is mapped directly via a standard MLP to match embedding_dim
        self.grasp_project = nn.Sequential(
            nn.Linear(12, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU()
        )

    def forward(self, pc, grasp, time):
        batch_size = pc.size(0)
        
        # 1. Time Embedding -> Shape: (B, embedding_dim)
        time = time.view(batch_size)
        time_emb = self.time_embedding(time) 
        
        # 2. Grasp Embedding -> Shape: (B, embedding_dim)
        # Direct MLP projection handles the 12 features globally and cleanly
        grasp_emb = self.grasp_project(grasp) 

        # 3. Context Fusion -> Shape: (B, 1, 2 * embedding_dim)
        ctx_emb = torch.cat([time_emb, grasp_emb], dim=-1).unsqueeze(1) 

        # 4. Denoising points
        out = pc
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        return out