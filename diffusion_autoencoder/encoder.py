import torch.nn as nn
import torch.nn.functional as F
import torch

 
class PointNetEncoder(nn.Module):
    def __init__(self, number_points, in_channels, hidden_channels, latent_dim, clamp):
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
    

class PointNetEncoder_paper(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super().__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v


class Encoder(nn.Module):
    def __init__(self, number_points=2048, in_channels=3, hidden_channels=64, latent_dim=32, clamp=False):
        super().__init__()
        self.encoder = PointNetEncoder(number_points, in_channels, hidden_channels, latent_dim, clamp)

    @staticmethod
    def KLD_loss(mean, log_variance):
        return -0.5 * torch.mean(1 + log_variance - mean.pow(2) - log_variance.exp())

    def forward(self, x):
        return self.encoder(x)

    def sample_latent_z(self, mean, log_variance):
        std = torch.exp(0.5 * log_variance) # compute std from log_variance
        eps = torch.randn_like(mean)
        return mean + eps * std