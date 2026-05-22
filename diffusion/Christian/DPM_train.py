import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from DPM_diffusion import *

def train_ae(dataloader:DataLoader, model:AutoEncoder, optimizer:Optimizer, writer):
    model.train()
    it = 0
    running_ae_loss = 0
    running_kld_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        ae_loss, encoder_loss = model.get_loss(batch)
        ae_loss.backward()
        optimizer.step()

        writer.add_scalar('Autoencoder Training Loss', ae_loss, it)
        writer.add_scalar("KLD Training Loss", encoder_loss, it)

        it += 1
        running_ae_loss += ae_loss.item()
        running_kld_loss += encoder_loss.item()
    return running_ae_loss, running_kld_loss


def train_decoder(dataloader:DataLoader, model:DiffusionPoint, encoder:PointNetEncoder, optimizer:Optimizer, writer):
    latent_list = list()
    model.train()
    encoder.eval()
    it = 0
    running_decoder_loss = 0
    for batch in dataloader:
        latent, _ = encoder(batch)
        latent_list.append(latent)
        
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        latent = latent_list[i]
        decoder_loss = model.get_loss(batch, latent)
        decoder_loss.backward()
        optimizer.step()

        writer.add_scalar('Decoder Training Loss', decoder_loss, it)

        it += 1
        running_decoder_loss += decoder_loss.item()

    return running_decoder_loss
