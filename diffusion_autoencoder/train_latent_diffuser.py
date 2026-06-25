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
from metrics import chamfer_distance
import trimesh
import random

import utils, dataset
from latent_decoder import Decoder
from latent_decoder import sample_ddim


def train(decoder, trainloader, valloader, device, optimizer, scheduler, config, model_config, writer_train, writer_val, debug_file):

    decoder.train()
    diffuser = decoder.diffuser
    denoiser = decoder.denoiser


    train_loss_running = 0.

    # For tracking loss per timestep
    timestep_loss_sum = defaultdict(float)
    timestep_counts = defaultdict(int)

    # Config parameters printed in tensorboard
    # config_markdown = "| Hyperparameter | Value |\n| :--- | :--- |\n"
    # for key, value in config.items():
    #     config_markdown += f"| **{key}** | {value} |\n"
    # writer_train.add_text("Hyperparameters", config_markdown, global_step=0)

    for epoch in range(model_config['last_epoch'], model_config['last_epoch'] + config['max_epochs']):
        
        train_loss_epoch_running = 0
        
        for i, batch in enumerate(trainloader):
            
            batch_grasp = batch["grasp"].to(device) #[B,12]
            batch_code = batch["code"].to(device)   #[B,128]

            optimizer.zero_grad()
            
            timestep_batch = torch.randint(0, diffuser.T, (batch_code.shape[0],), device=device).long() #[B,]
            
            noisy_code, actual_noise = diffuser.add_noise(batch_code, timestep_batch)
            predicted_noise = denoiser(noisy_code, timestep_batch, batch_grasp)

            mse_loss_per_item = F.mse_loss(predicted_noise, actual_noise, reduction='none').mean(dim=[1]) #[B]
            MSE_loss = mse_loss_per_item.mean()

            loss = MSE_loss


            loss.backward()
            optimizer.step()
            # scheduler.step()

            # print(f"Allocated pool: {torch.cuda.memory_reserved() / 1024**2:.2f} MiB") #DEBUG
            # print(f"True tensor usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB") #DEBUG

            # Loss logging
            train_loss_epoch_running += loss.item()
            train_loss_running += loss.item()

            #DEBUG
            # debug_file.write('train_loss_epoch_running: ' + str(train_loss_epoch_running) + '\n')
            # debug_file.flush()

            iteration = epoch * len(trainloader) + i

            # Train Loss
            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:03d}/{i:03d}] train_loss: {train_loss_running / config["print_every_n"]:.3f}')
                writer_train.add_scalar("Loss", train_loss_running / config["print_every_n"], iteration*config['train_batch_size'])
                train_loss_running = 0.

            # Average loss per timestep     
            for t_val, item_mse in zip(timestep_batch.tolist(), mse_loss_per_item.tolist()):
                timestep_loss_sum[t_val] += item_mse
                timestep_counts[t_val] += 1

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
        # print(f"[{epoch:03d}] integral: {MSE_per_timestep_integral:.05}")
        if MSE_per_timestep_integral < model_config['best_integral']:
                model_config['best_integral'] = MSE_per_timestep_integral
                utils.save_model(decoder, optimizer, scheduler, config, model_config, type = 'best_integral')
                # print(f'Best  Model:  {best_chamfer_loss:.05f}')

        # Clear dictionaries so the next epoch tracks completely fresh averages
        timestep_loss_sum.clear()
        timestep_counts.clear()

        # Weight plotting
        # for name, weight in encoder.named_parameters():
        #     writer_train.add_histogram(f"Weights/{name}", weight, epoch)
        #     if weight.grad is not None:
        #       writer_train.add_histogram(f"Gradients/{name}", weight.grad, epoch)
        # Weight plotting
        # for name, weight in decoder.named_parameters():
        #     writer_train.add_histogram(f"Weights/{name}", weight, epoch)
        #     if weight.grad is not None:
        #       writer_train.add_histogram(f"Gradients/{name}", weight.grad, epoch)

        # VALIDATION
        #___________________________________________________________________________________________________
        if epoch % config['validate_every_n_epochs'] == (config['validate_every_n_epochs'] - 1):
            
            # For tracking loss per timestep
            val_timestep_loss_sum = defaultdict(float)
            val_timestep_counts = defaultdict(int)

            decoder.eval()

            loss_total_val = 0
            for batch_val in valloader:
                batch_grasp = batch_val["grasp"].to(device) #[B,12]
                batch_code = batch_val["code"].to(device)   #[B,128]

                timestep_batch = torch.randint(0, diffuser.T, (batch_code.shape[0],), device=device).long() #[B,]
                
                with torch.no_grad():
                    noisy_code, actual_noise = diffuser.add_noise(batch_code, timestep_batch)
                    predicted_noise = denoiser(noisy_code, timestep_batch, batch_grasp)

                mse_loss_per_item = F.mse_loss(predicted_noise, actual_noise, reduction='none').mean(dim=[1]) #[B]
                MSE_loss = mse_loss_per_item.mean()

                loss = MSE_loss

                loss_total_val += loss.item()
                
                # Average loss per timestep     
                for t_val, item_mse in zip(timestep_batch.tolist(), mse_loss_per_item.tolist()):
                    val_timestep_loss_sum[t_val] += item_mse
                    val_timestep_counts[t_val] += 1
            
            # Val Loss
            loss_val = loss_total_val / len(valloader)
            print(f'[{epoch:03d}] val_loss: {loss_val:.3f}')
            writer_val.add_scalar("Loss", loss_val, iteration*config['train_batch_size'])
            

            avg_error_list = []
            valid_timesteps = []
            for time_val in range(diffuser.T):
                if val_timestep_counts[time_val] > 0:
                    avg_error = val_timestep_loss_sum[time_val] / val_timestep_counts[time_val]
                    avg_error_list.append(avg_error)
                    valid_timesteps.append(time_val)
                    writer_val.add_scalar(f"Val MSE Timestep Error/Epoch_{epoch:03d}", avg_error, global_step=time_val)


            if loss_val < model_config['best_val_loss']:
                model_config['best_val_loss'] = loss_val
                utils.save_model(decoder, optimizer, scheduler, config, model_config, type = 'best_val')
                # print(f"Best val Model:  {model_config['best_val_loss']:.05f}")

            decoder.train()

            # Clear dictionaries so the next epoch tracks completely fresh averages
            val_timestep_loss_sum.clear()
            val_timestep_counts.clear()

        # VALIDATION
        #___________________________________________________________________________________________________


        train_loss_epoch = train_loss_epoch_running / len(trainloader)
        #DEBUG
        # debug_file.write('train_loss_epoch: ' + str(train_loss_epoch) + '\n')
        # debug_file.flush()
        #DEBUG

        # Save model weights if the best loss is achieved  
        if  train_loss_epoch < model_config['best_train_loss']:
            model_config['best_train_loss'] = train_loss_epoch
            utils.save_model(decoder, optimizer, scheduler, config, model_config, type = "best_train")
            print(f"Best train Model: {model_config['best_train_loss']:.03f}")
            #DEBUG
            # debug_file.write('best_train_loss: ' + str(best_train_loss) + '\n')
            # debug_file.flush()
            #DEBUG

        
        # Checkpoint saving
        model_config['last_epoch'] = epoch
        utils.save_model(decoder, optimizer, scheduler, config, model_config, type = 'checkpoint')


    print('Best Loss: ', model_config['best_train_loss'])
    print('Best Val Loss: ', model_config['best_val_loss'])
    print('Best Integral:  ', model_config['best_integral'])






if __name__ == "__main__":

    # Model name
    current_time = datetime.datetime.now().strftime("%b%d_%H-%M")
    denoiser_name = "epochs=1000_latent_dim=128_hidden_dim=128_embedding_dim=120"
    experiment_name = f"{current_time}_{denoiser_name}"


    config = {
        'experiment_name': experiment_name,
        'device': 'cuda:0',
        'train_batch_size': 64,
        'val_batch_size': 128,
        'learning_rate': 0.0004,
        'step_size': 10, # scheduler step, one step is one batch
        'gamma': 1,
        'max_epochs': 1000,
        'timesteps': 1000,
        'print_every_n': 30, # every n batches
        'validate_every_n_epochs': 15,
    }

    model_config = {
        'last_epoch': 0,
        'latent_dim': 128,
        'hidden_dim': 128,
        'embedding_dim': 120,
        'best_train_loss': 100,
        'best_chamfer_loss': 100,
        'best_val_loss': 100,
        'best_integral':1000
    }

    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        device = torch.device('cpu')
        print('Using CPU')

    num_workers = max(1, os.cpu_count() - 1)
    print(num_workers)
    trainset = dataset.Dataset_Latent_grasp_and_code('train', config['timesteps'], fake=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['train_batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True)
    valset = dataset.Dataset_Latent_grasp_and_code('val', config['timesteps'], fake=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=config['val_batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True)

    decoder = Decoder(
        latent_dim=128, hidden_dim=128, embedding_dim=120,  
        timesteps=config['timesteps'], beta_start=1e-4, beta_end=0.02)

    decoder.to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['step_size'], config['gamma'])


    total_dec, trainable_dec = utils.count_parameters(decoder)
    print(f"Decoder Total: {total_dec:,} | Trainable: {trainable_dec:,} | Model size: {utils.model_memory_size(decoder):.3f} MB")

    torch.cuda.empty_cache()
    #tensorboard --logdir=logs
    
    # #Reload model
    # experiment_name = "Jun16_17-40_epochs=1000 latent_dim=128 hidden_dim=128 embedding_dim=120"
    # config, model_config, decoder = utils.reload_model(optimizer, scheduler, experiment_name, 'checkpoint', device)
    # config['max_epochs'] = 2000 - model_config['last_epoch']
    # print(f"Checkpoint loaded! {config['max_epochs']} more epochs to go!")
    # experiment_name = "Jun16_17-40_100_Objects_2000epochs_64latent_enc128_dec256_globalnorm"
    # config['experiment_name'] = experiment_name

    train_log_path = pathlib.Path(f"logs/{datetime.datetime.now().strftime('%b%d')}/{config['experiment_name']}/train")
    writer_train = SummaryWriter(train_log_path)
    val_log_path = pathlib.Path(f"logs/{datetime.datetime.now().strftime('%b%d')}/{config['experiment_name']}/val")
    writer_val = SummaryWriter(val_log_path)

    with open("debug.txt", "w", encoding="utf-8") as debug_file:
        train(
            decoder, trainloader, valloader, device, optimizer, scheduler, 
            config, model_config, writer_train, writer_val, debug_file)

    writer_train.close()
    writer_val.close()