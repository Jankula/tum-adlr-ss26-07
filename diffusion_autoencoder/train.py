import pathlib
import os
import datetime
from collections import defaultdict
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from scipy.integrate import trapezoid

import utils, dataset
from encoder import Encoder
from decoder import Decoder


def train(encoder, decoder, trainloader, valloader, device, optimizer, scheduler, config, model_config, writer_train, writer_val, debug_file):

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

            timestep_batch = torch.randint(0, diffuser.T, (batch.shape[0],), device=device).long() #[B,]
            # beta_batch = diffuser.beta[timestep_batch]
            
            noisy_pc, actual_noise = diffuser.add_noise(batch, timestep_batch)
            predicted_noise = denoiser(noisy_pc, timestep_batch, code)

            # Loss calculation
            KLD_loss = encoder.KLD_loss(mean, log_variance)

            mse_loss_per_item = F.mse_loss(predicted_noise, actual_noise, reduction='none').mean(dim=[1, 2]) #[B]
            MSE_loss = mse_loss_per_item.mean()

            # loss = MSE_loss + min(0.01, 0.0001 * 1.2 ** epoch) * KLD_loss
            loss = MSE_loss + 0.001 * KLD_loss


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
            #             mean, log_variance = encoder(batch_val)
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
        avg_error_list = []
        valid_timesteps = []
        for time_val in range(diffuser.T):
            if timestep_counts[time_val] > 0:
                avg_error = timestep_loss_sum[time_val] / timestep_counts[time_val]
                avg_error_list.append(avg_error)
                valid_timesteps.append(time_val)
                writer_train.add_scalar(f"MSE Timestep Error/Epoch_{epoch:03d}", avg_error, global_step=time_val)
        
        # Calculate the integral of the average mse per timestep
        MSE_per_timestep_integral = trapezoid(np.array(avg_error_list), np.array(valid_timesteps))
        writer_train.add_scalar(f"MSE per Timestep Integral", MSE_per_timestep_integral, epoch)
        print("MSE per timestep integral: ", MSE_per_timestep_integral)

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

if __name__ == "__main__":

    # Model name
    current_time = datetime.datetime.now().strftime("%b%d_%H-%M")
    denoiser_name = "10_Objects_1000timestep_500epochs_6latent_16batch_KL0.001_hidden_dec128"
    experiment_name = f"{current_time}_{denoiser_name}"


    config = {
        'experiment_name': experiment_name,
        'device': 'cuda:0',
        'batch_size': 16,
        'resume_ckpt': None,
        'learning_rate': 0.0004,
        'step_size': 10, # scheduler step, one step is one batch
        'gamma': 1,
        'max_epochs': 500,
        'timesteps': 1000,
        'print_every_n': 30, # every n batches
        # 'validate_every_n': 10,
    }

    model_config = {
        'last_epoch': 0,
        'enc_hidden_channels': 64,
        'dec_hidden_dim': 128,
        'latent_dim': 32
    }

    # declare device
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        device = torch.device('cpu')
        print('Using CPU')

    # create dataloaders
    trainset = dataset.Dataset('train', config['timesteps'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    valset = dataset.Dataset('val', config['timesteps'])
    valloader = torch.utils.data.DataLoader(valset, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    encoder = Encoder(number_points=2048, in_channels=3, hidden_channels=model_config['enc_hidden_channels'], latent_dim=model_config['latent_dim'], clamp=False)
    decoder = Decoder(number_points=2048, point_dim=3, hidden_dim=model_config['dec_hidden_dim'], latent_dim=model_config['latent_dim'], timesteps=config['timesteps'], beta_start=1e-4, beta_end=0.02)

    # move model to specified device
    encoder.to(device)
    decoder.to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['step_size'], config['gamma'])
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0006, steps_per_epoch=len(trainloader), epochs=config['max_epochs'], anneal_strategy='cos', three_phase=True)


    total_enc, trainable_enc = utils.count_parameters(encoder)
    total_dec, trainable_dec = utils.count_parameters(decoder)
    print(f"Encoder Total: {total_enc:,} | Trainable: {trainable_enc:,} | Model size: {utils.model_memory_size(encoder):.3f} MB")
    print(f"Decoder Total: {total_dec:,} | Trainable: {trainable_dec:,} | Model size: {utils.model_memory_size(decoder):.3f} MB")

    # start training
    torch.cuda.empty_cache()
    #tensorboard --logdir=diffusion_autoencoder/logs
    # Create tensorboard writer    
    log_path = pathlib.Path(f"logs/{datetime.datetime.now().strftime('%b%d')}/{config['experiment_name']}")
    writer = SummaryWriter(log_path)

    with open("debug.txt", "w", encoding="utf-8") as debug_file:
        train.train(encoder, decoder, trainloader, None, device, optimizer, scheduler, config, model_config, writer, None, debug_file)

    writer.close()
