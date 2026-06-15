import pathlib
import os
from collections import defaultdict
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter

import network, utils, dataset
from decoder import *


def main(config):
    """
    Driver function for training a diffusion model
    :param config: configuration for training - has the following keys
                   'experiment_name': name of the experiment, checkpoint will be saved to folder "models/<experiment_name>"
                   'device': device on which model is trained, e.g. 'cpu' or 'cuda:0'
                   'resume_ckpt': None if training from scratch, otherwise path to checkpoint (saved weights)
                   'learning_rate': learning rate for optimizer
                   'timesteps' : the number of timesteps for the diffusion process
                   'max_epochs': total number of epochs after which training should stop
                   'batch_size': batch size for training and validation dataloaders
                   'print_every_n': print train loss every n iterations
                   'validate_every_n': print validation loss and validation accuracy every n iterations
                   'is_overfit': if the training is done on a small subset of data specified in exercise_2/split/overfit.txt,
                                 train and validation done on the same set, so error close to 0 means a good overfit. Useful for debugging.
    """

    # declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    # create dataloaders
    trainset = dataset.Dataset('train' if not config['is_overfit'] else 'overfit')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    valset = dataset.OverfitDataset('val' if not config['is_overfit'] else 'overfit')
    valloader = torch.utils.data.DataLoader(valset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    denoiser = network.Denoiser()
    diffuser = network.Diffuser(config['timesteps'])

    # load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        utils.reload_model(denoiser, diffuser, config['experiment_name'], device)

    # move model to specified device
    denoiser.to(device)
    diffuser.to(device)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=config['learning_rate'])

    # Create tensorboard writer    
    log_path = pathlib.Path(f"logs/diffusion_training/{config['experiment_name']}")
    writer = SummaryWriter(log_path)

    #Run this code in terminal to start tensorboard: tensorboard --logdir=diffusion/nikola/logs/diffusion_training


    total, trainable = utils.count_parameters(denoiser)
    print(f"Total: {total:,} | Trainable: {trainable:,} | Model size: {utils.model_memory_size(denoiser):.3f} MB")

    # start training
    train(denoiser=denoiser, diffuser=diffuser, trainloader=trainloader, device=device, optimizer=optimizer, config=config, writer=writer,valloader=None)


def train(encoder, decoder, trainloader, valloader, device, optimizer, scheduler, config, model_config, writer_train, writer_val, debug_file):

    optimizer = optimizer
    # scheduler = scheduler

    encoder.train()
    decoder.train()
    diffuser = decoder.diffuser
    denoiser = decoder.denoiser


    best_train_loss = 10
    best_chamfer_loss = 10
    best_val_loss = 10

    train_loss_running = 0.

    # For tracking loss per timestep
    timestep_loss_sum = defaultdict(float)
    timestep_counts = defaultdict(int)

    # Config parameters printed in tensorboard
    config_markdown = "| Hyperparameter | Value |\n| :--- | :--- |\n"
    for key, value in config.items():
        config_markdown += f"| **{key}** | {value} |\n"
    writer_train.add_text("Hyperparameters", config_markdown, global_step=0)

    for epoch in range(model_config['last_epoch'], model_config['last_epoch'] + config['max_epochs']):
        
        train_loss_epoch_running = 0
        
        for i, batch in enumerate(trainloader):
            # print("Batch shape: " + str(batch.shape))#DEBUG
            
            batch = batch.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            mean, log_variance = encoder(batch)
            code = encoder.sample_latent_z(mean, log_variance)

            timestep_batch = torch.randint(0, diffuser.T, (batch.shape[0],), device=device).long()
            beta_batch = diffuser.beta[timestep_batch]
            
            noisy_pc, actual_noise = diffuser.add_noise(batch, timestep_batch)
            predicted_noise = denoiser(noisy_pc, beta_batch, code)

            # Loss calculation
            KLD_loss = encoder.KLD_loss(mean, log_variance)

            mse_loss_per_item = F.mse_loss(predicted_noise, actual_noise, reduction='none').mean(dim=[1, 2]) #[B]
            MSE_loss = mse_loss_per_item.mean()

            loss = MSE_loss + min(0.01, 0.0001 * 1.2 ** epoch) * KLD_loss


            loss.backward()
            optimizer.step()
            # scheduler.step()

            # print(f"Allocated pool: {torch.cuda.memory_reserved() / 1024**2:.2f} MiB") #DEBUG
            # print(f"True tensor usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB") #DEBUG

            # Loss logging
            train_loss_epoch_running += loss.item()
            train_loss_running += loss.item()

            #DEBUG
            debug_file.write('train_loss_epoch_running: ' + str(train_loss_epoch_running) + '\n')
            debug_file.flush()

            iteration = epoch * len(trainloader) + i

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:02d}/{i:03d}] train_loss: {train_loss_running / config["print_every_n"]:.3f}')
                writer_train.add_scalar("Train Loss", train_loss_running / config["print_every_n"], iteration*config['batch_size'])
                train_loss_running = 0.

            # Average loss per timestep     
            for t_val, item_mse in zip(timestep_batch.tolist(), mse_loss_per_item.tolist()):
                timestep_loss_sum[t_val] += item_mse
                timestep_counts[t_val] += 1

            # # validation evaluation and logging
            # if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):

            #     # set model to eval, important if your network has e.g. dropout or batchnorm layers
            #     encoder.eval()
            #     decoder.eval()

            #     loss_total_val = 0
            #     # forward pass and evaluation for entire validation set
            #     for batch_val in valloader:
            #         batch_val = batch_val.to(device)

            #         with torch.no_grad():
            #             mean, log_variance = encoder(batch)
            #             code = encoder.sample_latent_z(mean, log_variance)

            #             timestep_batch = torch.randint(0, diffuser.T, (batch.shape[0],), device=device).long()
            #             beta_batch = diffuser.beta[timestep_batch]
                        
            #             noisy_pc, actual_noise = diffuser.add_noise(batch, timestep_batch)
            #             predicted_noise = denoiser(noisy_pc, beta_batch, code)

            #             # Loss calculation
            #             KLD_loss = encoder.KLD_loss(mean, log_variance)

            #             mse_loss_per_item = F.mse_loss(predicted_noise, actual_noise, reduction='none').mean(dim=[1, 2]) #[B]
            #             MSE_loss = mse_loss_per_item.mean()

            #             loss = MSE_loss + 0.01 * KLD_loss 

            #         loss_total_val += loss.item()
                
            #     loss_val = loss_total_val / len(valloader)
            #     print(f'[{epoch:03d}/{i:04d}] val_loss: {loss_val:.3f}')
            #     writer_val.add_scalar("Val Loss", loss_val, iteration*config['batch_size'])

            #     if loss_val < best_val_loss:
            #         utils.save_model(encoder, decoder, optimizer, scheduler, config, model_config, type = 'best')
            #         best_val_loss = loss_val

            #     # set model back to train
            #     encoder.train()
            #     decoder.train()
            

        # Average loss per timestep            
        for time_val in range(diffuser.T):
            if timestep_counts[time_val] > 0:
                avg_error = timestep_loss_sum[time_val] / timestep_counts[time_val]
                writer_train.add_scalar(f"MSE Timestep Error/Epoch_{epoch:03d}", avg_error, global_step=time_val)
                
        # Clear dictionaries so the next epoch tracks completely fresh averages
        timestep_loss_sum.clear()
        timestep_counts.clear()

        # Weight plotting
        for name, weight in encoder.named_parameters():
            writer_train.add_histogram(f"Weights/{name}", weight, epoch)
            if weight.grad is not None:
              writer_train.add_histogram(f"Gradients/{name}", weight.grad, epoch)
        # Weight plotting
        for name, weight in decoder.named_parameters():
            writer_train.add_histogram(f"Weights/{name}", weight, epoch)
            if weight.grad is not None:
              writer_train.add_histogram(f"Gradients/{name}", weight.grad, epoch)


        train_loss_epoch = train_loss_epoch_running / len(trainloader)
        #DEBUG
        debug_file.write('train_loss_epoch: ' + str(train_loss_epoch) + '\n')
        debug_file.flush()
        #DEBUG
        # Save model weights if the best loss is achieved  
        if  train_loss_epoch < best_train_loss:
            utils.save_model(encoder, decoder, optimizer, scheduler, config, model_config, type = "best")
            best_train_loss = train_loss_epoch
            print('Best Model:  ', best_train_loss)
            #DEBUG
            debug_file.write('best_train_loss: ' + str(best_train_loss) + '\n')
            debug_file.flush()
            #DEBUG

        
        # Model epoch saving
        # model_config['last_epoch'] = epoch
        # utils.save_model(encoder, decoder, optimizer, scheduler, config, model_config, type = 'epoch' + str(epoch))

        # visualization.plot_interactive_epochs(config)



    print('Best Loss: ', best_train_loss)
    print('Best Val Loss: ', best_val_loss)

def train_ae(train_dataloader:DataLoader, model:AutoEncoder, optimizer:Optimizer, writer):
    model.train()
    it = 0
    running_ae_loss = 0
    running_kld_loss = 0
    for batch in train_dataloader:
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
