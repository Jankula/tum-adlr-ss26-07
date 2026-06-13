import os
import torch
import numpy as np
import pathlib
from encoder import Encoder
from decoder import Decoder


def save_model(encoder, decoder, optimizer, scheduler, config, model_config, type):
    checkpoint = {
    'encoder_state': encoder.state_dict(),
    'decoder_state': decoder.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'scheduler_state': scheduler.state_dict(),
    'config': config,
    'model_config': model_config
    }
    path = pathlib.Path(f'models/{config["experiment_name"]}')
    path.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, pathlib.Path(f'models/{config["experiment_name"]}/{type}.pt'))

def reload_model(optimizer, scheduler, experiment_name, type, device):
    checkpoint = torch.load(pathlib.Path(f'models/{experiment_name}/{type}.pt'), weights_only=True, map_location=device)

    config = checkpoint['config']
    
    model_config = checkpoint['model_config']

    encoder = Encoder(number_points=2048, in_channels=3, hidden_channels=model_config['enc_hidden_channels'], latent_dim=model_config['latent_dim'], clamp=False)
    decoder = Decoder(number_points=2048, point_dim=3, hidden_dim=model_config['dec_hidden_dim'], latent_dim=model_config['latent_dim'], timesteps=config['timesteps'], beta_start=1e-4, beta_end=0.02)

    encoder.to(device)
    decoder.to(device)

    encoder.load_state_dict(checkpoint['encoder_state'])
    decoder.load_state_dict(checkpoint['decoder_state'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state'])

    encoder.eval()
    decoder.eval()

    return config, model_config, encoder, decoder


def count_parameters(model):
    # Total parameters (including non-trainable ones)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Only parameters that require gradients
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def model_memory_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb