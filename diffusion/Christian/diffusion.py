import torch



class DiffusionSchedule:
    def __init__(self, timesteps, device, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)

        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat)

        alpha_hat_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_hat[:-1]], dim=0)
        self.posterior_variance = self.betas * (1.0 - alpha_hat_prev) / (1 - self.alpha_hat)
        self.posterior_variance2 = self.betas

    def extract(self, arr, t, x_shape):
        out = arr.gather(0, t)
        
        return out.view(-1, 1, 1).expand(x_shape)
    
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_hat = self.extract(self.sqrt_alpha_hat, t, x0.shape)
        sqrt_one_minus_alpha_hat = self.extract(self.sqrt_one_minus_alpha_hat, t, x0.shape)
        xt = sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise 
        
        return xt, noise


    