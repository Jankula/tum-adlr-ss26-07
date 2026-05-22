import torch
import torch.nn
from pytorch3d.loss import chamfer_distance
import numpy as np
np.bool8 = np.bool
from torch.utils.tensorboard.writer import SummaryWriter

def KLD_loss(mean, log_variance):
    return -0.5 * torch.mean(1 + log_variance - mean.pow(2) - log_variance.exp())

def linear_annealing(total_epochs, current_epoch, beta_start, beta_target):
    """linearly increases beta for KL annealing -> smoothly transition latent space into gaussian shape"""
    if current_epoch > total_epochs: # beta can not be greater than beta_target
        current_epoch = total_epochs
    return beta_start + current_epoch * (beta_target - beta_start) / total_epochs


def train_one_epoch(model, dataloader, optimizer, epochs, epoch, writer, beta_start=1e-6, beta_target=1e-2):
    running_loss = 0
    chamfer_loss = 0
    kld_loss_total = 0
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        reconstruction, mean, log_variance = model(batch)
        reconstruction_loss, _ = chamfer_distance(reconstruction, batch)
        #print(type(reconstruction_loss))
        #print(reconstruction_loss.shape)
        #print(type(kld_loss))
        #print(kld_loss.shape)
        kld_loss = KLD_loss(mean, log_variance)
        beta = linear_annealing(epochs, epoch, beta_start, beta_target)
        vae_loss = reconstruction_loss + beta * kld_loss

        iteration = epoch * len(dataloader) + i
    
        writer.add_scalar("VAE Training Loss", vae_loss, iteration)
        writer.add_scalar("KLD Training Loss", kld_loss, iteration)
        writer.add_scalar("Chamfer Training Loss", reconstruction_loss, iteration)


        vae_loss.backward()
        optimizer.step()

    running_loss += vae_loss.item()
    kld_loss_total += kld_loss.item()
    chamfer_loss = reconstruction_loss.item()

    return running_loss, chamfer_loss, kld_loss_total

