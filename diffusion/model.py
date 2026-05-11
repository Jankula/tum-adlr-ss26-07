import torch.nn as nn
import torch
import math

def sinusoidal_timestep_embedding(timesteps, dim=256, max_period=10000):
    half = dim // 2
    frequs = torch.exp(-math.log(max_period) * torch.arange(0, half, device="cpu").float() / half)
    args = timesteps[None, :].float() * frequs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb

class SimpleMLP(nn.Module):
    def __init__(self, t_dim, out_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, t_dim):
        super().__init__()

        self.time_embedding = nn.Linear(t_dim, hidden_channels)

        
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, 1)
        self.norm1 = nn.GroupNorm(8, hidden_channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(hidden_channels, out_channels, 1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        

        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_embedding(t_emb)[:, :, None]
        return nn.functional.silu(self.norm2(self.conv2(h)) + self.skip(x))
        

    
class ResNet(nn.Module):
    def __init__(self, number_points=2048, in_channels=3, hidden_channels=64, out_channels=3, t_dim=256):
        super().__init__()
        self.input_layer == nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, 2 * hidden_channels),
            nn.GroupNorm(8, 2 * hidden_channels),
            nn.SiLU(),
            nn.Conv1d(2 * hidden_channels, 4 * hidden_channels),
            nn.GroupNorm(8, 4 * hidden_channels),
            nn.SiLU(),
        )
        
        self.max_pool1 = nn.MaxPool1d(number_points)
        self.res_block1 = ResBlock(4 * hidden_channels, 4 * hidden_channels)
        self.max_pool2 = nn.MaxPool1d(number_points)
        self.res_block2 = ResBlock(8 * hidden_channels, 8 * hidden_channels)
        self.downsample2 = nn.Conv1d(8 * hidden_channels, 4 * hidden_channels)
        self.max_pool3 = nn.MaxPool1d(number_points)
        self.res_block3 = ResBlock(8 * hidden_channels, 4 * hidden_channels)
        self.downsample3 = nn.Conv1d(4 * hidden_channels, 2 * hidden_channels, 1)
        self.norm1 = nn.GroupNorm(8, 2 * hidden_channels)
        self.act = nn.SiLU()
        self.downsample4 = nn.Conv1d(2 * hidden_channels, hidden_channels, 1)
        self.norm2 = nn.GroupNorm(8, hidden_channels)
        self.output = nn.Conv1d(hidden_channels, out_channels, 1)
        self.norm3 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.Tanh()
    
    def forward(self, x, t_emb):
        pass
        

