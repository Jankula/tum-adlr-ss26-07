import torch
import diffusion
import DataPipeline
from torch.utils.data import DataLoader
import model
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def train_one_epoch(model, loader, diffusion, optimizer, device, grad_clip=1.0, permute=False):
    model.train()
    running_loss = 0
    step1000_loss = []
    step800_loss = []
    step600_loss = []
    step400_loss = []
    step200_loss = []
    step100_loss = []
    loss_1000 = 0
    loss_800 = 0
    loss_600 = 0
    loss_400 = 0
    loss_200 = 0
    loss_100 = 0
    
    for batch in loader:
        batch = batch.to(device)
        if permute == True:
            batch = batch.permute(0, 2, 1)
        batch_size = batch.size(0)
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)
        x_t, noise = diffusion.q_sample(batch, t)

        optimizer.zero_grad(set_to_none=True)

        pred_noise = model(x_t, t)

        for i, step in enumerate(t):
            if step > 800 and step <= 1000:
                step1000_loss.append(torch.nn.functional.mse_loss(pred_noise[i], noise[i]).item())
            elif step > 600 and step <= 800:
                step800_loss.append(torch.nn.functional.mse_loss(pred_noise[i], noise[i]).item())
            elif step > 400 and step <= 600:
                step600_loss.append(torch.nn.functional.mse_loss(pred_noise[i], noise[i]).item())
            elif step > 200 and step <= 400:
                step400_loss.append(torch.nn.functional.mse_loss(pred_noise[i], noise[i]).item())
            elif step > 100 and step <= 200:
                step200_loss.append(torch.nn.functional.mse_loss(pred_noise[i], noise[i]).item())
            elif step >= 0 and step <= 100:
                step100_loss.append(torch.nn.functional.mse_loss(pred_noise[i], noise[i]).item())

        loss = torch.nn.functional.mse_loss(pred_noise, noise)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()

        running_loss += loss.item()
    if len(step1000_loss) != 0:
        loss_1000 = sum(step1000_loss) / len(step1000_loss)
    if len(step800_loss) != 0:
        loss_800 = sum(step800_loss) / len(step800_loss)
    if len(step600_loss) != 0:
        loss_600 = sum(step600_loss) / len(step600_loss)
    if len(step400_loss) != 0:
        loss_400 = sum(step400_loss) / len(step400_loss)
    if len(step200_loss) != 0:
        loss_200 = sum(step200_loss) / len(step200_loss)
    if len(step100_loss) != 0:
        loss_100 = sum(step100_loss) / len(step100_loss)

    return (running_loss / max(len(loader), 1), loss_1000, loss_800, loss_600, loss_400, loss_200, loss_100)

@torch.no_grad
def p_sample(model, diffusion, x, t):
    alphas_t = diffusion.extract(diffusion.alphas, t, x.shape)
    sqrt_one_minus_alpha_hat_t = diffusion.extract(diffusion.sqrt_one_minus_alpha_hat, t, x.shape)
    pred_noise = model(x, t)

    model_mean = (1.0 / torch.sqrt(alphas_t)) * (
        x - ((1.0 - alphas_t) / sqrt_one_minus_alpha_hat_t) * pred_noise
    )

    posterior_var_t = diffusion.extract(diffusion.posterior_variance, t, x.shape)
    posterior_var_t2 = diffusion.extract(diffusion.posterior_variance2, t, x.shape)
    noise = torch.randn_like(x)
    #noise = torch.zeros_like(x)

    nonzero_mask = (t != 0).float().view(-1, 1, 1)
    #x_prev = model_mean + nonzero_mask * torch.sqrt(torch.clamp(posterior_var_t, min=1e-20)) * noise
    x_prev = model_mean + nonzero_mask * torch.sqrt(posterior_var_t2) * noise

    return x_prev

@torch.no_grad
def sample_point_clouds(model, diffusion, n=1, size=2048, channels=3, device="cpu", permute=False):
    x_list = []
    model.eval()
    x = torch.randn(n, channels, size, device=device)
    if permute == True:
        x = x.permute(0, 2, 1)

    for timestep in reversed(range(diffusion.timesteps)):
        t = torch.full((n,), timestep, device=device, dtype=torch.long)
        x = p_sample(model, diffusion, x, t) 
        x_list.append(x.detach().cpu().permute(0, 2, 1).numpy())

    return x_list

