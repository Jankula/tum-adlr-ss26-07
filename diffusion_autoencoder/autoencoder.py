import torch.nn as nn
import torch.nn.functional as F
import torch

from encoder import *
from decoder import *

class AutoEncoder(nn.Module):

    def __init__(self, number_points=2048, point_dim=3, hidden_dim=64, latent_dim=32, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.encoder = Encoder(number_points, point_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(number_points, point_dim, hidden_dim, latent_dim, timesteps, beta_start, beta_end)
            

    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, 3).
        """
        mean, log_variance = self.encoder(x)
        
        return self.encoder.sample_latent_z(mean, log_variance), mean, log_variance

    def decode(self, code, num_points):
        return self.diffusion.sample(num_points, code, flexibility=flexibility, ret_traj=ret_traj)

    def get_loss(self, x):
        code, mean, log_variance = self.encode(x)
        encoder_loss = 0.01 * KLD_loss(mean, log_variance)
        denoiser_loss = self.diffusion.get_loss(x, code)
        return denoiser_loss + encoder_loss, encoder_loss