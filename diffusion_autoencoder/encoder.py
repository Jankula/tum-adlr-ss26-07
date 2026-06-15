import torch.nn as nn
import torch.nn.functional as F
import torch

 
class Resnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size),
        )

        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()

    def forward(self, x):
        return F.relu(self.resblock(x) + self.skip(x))
    

class PointNetEncoder(nn.Module):
    def __init__(self, number_points=2048, in_channels=3, hidden_channels=128, latent_dim=256, clamp=False):
        super().__init__()
        self.number_points = number_points
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.clamp = clamp

        #Shared MLP idea from Pointnet with Conv1d, does not weight order of points in pointcloud
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1), #Conv1d expects [B, C, N]
            nn.GroupNorm(8, hidden_channels), #GroupNorm for small batch_size training, Batchnorm weak to noisy batch statistics
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 2 * hidden_channels, 1),
            nn.GroupNorm(8, 2 * hidden_channels),
            nn.ReLU(),
            nn.Conv1d(2 * hidden_channels, 4 * hidden_channels, 1),
        )

        self.max_pool = nn.MaxPool1d(kernel_size=number_points)

        #For calculating the mean of the latent variable
        self.fc1_mean = nn.Linear(4 * hidden_channels, 2 * hidden_channels)
        self.fc2_mean = nn.Linear(2 * hidden_channels, hidden_channels)
        self.fc3_mean = nn.Linear(hidden_channels, latent_dim)

        self.fc1_mean_norm = nn.LayerNorm(2 * hidden_channels)
        self.fc2_mean_norm = nn.LayerNorm(hidden_channels)
        # self.fc3_mean_norm = nn.LayerNorm(latent_dim)

        #For calculating the log-variance of the latent variable
        self.fc1_var = nn.Linear(4 * hidden_channels, 2 * hidden_channels)
        self.fc2_var = nn.Linear(2 * hidden_channels, hidden_channels)
        self.fc3_var = nn.Linear(hidden_channels, latent_dim)

        self.fc1_var_norm = nn.LayerNorm(2 * hidden_channels)
        self.fc2_var_norm = nn.LayerNorm(hidden_channels)
        # self.fc3_var_norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        x = x.transpose(1, 2) # Transform from [B, N, C] to [B, C, N]
        B, C, N = x.shape

        x = self.projection(x)
        x = self.max_pool(x)
        x = x.view(-1, 4 * self.hidden_channels)

        #Reparametrization trick, to enable backpropagation training of the VAE
        #Model learns mean and log_variance of the Gaussian latent space distribution
        #Sampling of latents handled via outside function with learned mean and log_variance
        mean = F.silu(self.fc1_mean_norm(self.fc1_mean(x)))
        mean = F.silu(self.fc2_mean_norm(self.fc2_mean(mean)))
        mean = self.fc3_mean(mean)

        log_variance = F.silu(self.fc1_var_norm(self.fc1_var(x)))
        log_variance = F.silu(self.fc2_var_norm(self.fc2_var(log_variance)))
        log_variance = self.fc3_var(log_variance)

        if self.clamp:
            log_variance = torch.clamp(log_variance, min=-30.0, max=20.0) # For numerical stability

        return mean, log_variance
    

class ResnetDecoder(nn.Module):
    def __init__(self, number_points, latent_dim):
        super().__init__()
        self.number_points = number_points
        self.latent_dim = latent_dim

        self.upsample1 = nn.Linear(latent_dim, 2 * latent_dim)
        self.norm1 = nn.LayerNorm(2 * latent_dim)
        self.resblock1 = Resnet(2 * latent_dim, 2 * latent_dim, 2 * latent_dim)
        self.upsample2 = nn.Linear(2 * latent_dim, 4 * latent_dim)
        self.norm2 = nn.LayerNorm(4 * latent_dim)
        self.resblock2 = Resnet(4 * latent_dim, 4 * latent_dim, 4 * latent_dim)
        self.output = nn.Linear(4 * latent_dim, number_points * 3)

    def forward(self, x):
        B, _ = x.shape
        x = F.relu(self.norm1(self.upsample1(x)))
        x = self.resblock1(x)
        x = F.relu(self.norm2(self.upsample2(x)))
        x = self.resblock2(x)
        return self.output(x).view(B, self.number_points, 3) #[B, N, C]
    
class PointnetVAE(nn.Module):
    def __init__(self, number_points=2048, in_channels=3, hidden_channels=64, latent_dim=256):
        super().__init__()
        self.encoder = PointNetEncoder(number_points, in_channels, hidden_channels, latent_dim, clamp=False)
        self.decoder = ResnetDecoder(number_points, latent_dim)

    def sample_latent_z(self, mean, log_variance):
        std = torch.exp(0.5 * log_variance) # compute std from log_variance
        eps = torch.randn_like(mean)
        return mean + eps * std # encoder latent z
    
    def forward(self, x):
        mean, log_variance = self.encoder(x)
        z = self.sample_latent_z(mean, log_variance)
        return self.decoder(z), mean, log_variance # Reconstructed Point Cloud + mean and log_variance for KLD loss