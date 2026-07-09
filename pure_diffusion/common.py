import torch.nn as nn
import torch.nn.functional as F
import torch


# class SineCosineEncoding(nn.Module):
#     """Encodes the integer timestep 't' into a vector."""
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, t):
#         device = t.device
#         half_dim = self.dim // 2
#         emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
#         emb = t[:, None] * emb[None, :]
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return emb

class SineCosineEncoding(nn.Module):
    """Encodes a tensor of values into sine/cosine embeddings."""
    def __init__(self, dim):
        super().__init__()
        # Dimension must be even for an equal split of sin and cos
        assert dim % 2 == 0, "Embedding dimension must be even."
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        
        # Compute the frequency bands
        # Shape: (half_dim,)
        inv_freq = torch.exp(
            torch.arange(half_dim, device=device).float() * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        
        # Reshape t and inv_freq to broadcast nicely together.
        # Adding an extra dimension at the very end of 't' allows it to 
        # multiply smoothly with 'inv_freq', no matter if 't' is 1D, 2D, etc.
        # t[..., None] shape: (*, 1) | inv_freq[None, :] shape: (1, half_dim)
        # Resulting shape: (*, half_dim)
        emb = t[..., None] * inv_freq[None, :]
        
        # Cat sine and cosine components along the last dimension
        # Resulting shape: (*, dim)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


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
    

def KLD_loss(mean, log_variance):
    return -0.5 * torch.mean(1 + log_variance - mean.pow(2) - log_variance.exp())

def linear_annealing(total_epochs, current_epoch, beta_start, beta_target):
    """linearly increases beta for KL annealing -> smoothly transition latent space into gaussian shape"""
    if current_epoch > total_epochs: # beta can not be greater than beta_target
        current_epoch = total_epochs
    return beta_start + current_epoch * (beta_target - beta_start) / total_epochs    