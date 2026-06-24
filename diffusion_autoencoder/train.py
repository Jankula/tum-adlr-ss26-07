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
from encoder import Encoder
from decoder import Decoder
from decoder import sample_ddim


def train(encoder, decoder, trainloader, valloader, device, optimizer, scheduler, config, model_config, writer_train, writer_val, debug_file):

    encoder.train()
    decoder.train()
    diffuser = decoder.diffuser
    denoiser = decoder.denoiser


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
            
            batch = batch["point_cloud"].to(device)

            optimizer.zero_grad()
            
            # Forward pass
            mean, log_variance = encoder(batch)
            code = encoder.sample_latent_z(mean, log_variance)

            timestep_batch = torch.randint(0, diffuser.T, (batch.shape[0],), device=device).long() #[B,]
            
            noisy_pc, actual_noise = diffuser.add_noise(batch, timestep_batch)
            predicted_noise = denoiser(noisy_pc, timestep_batch, code)

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
                utils.save_model(encoder, decoder, optimizer, scheduler, config, model_config, type = 'best_integral')
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

            encoder.eval()
            decoder.eval()

            loss_total_val = 0
            for batch_val in valloader:
                batch_val = batch_val["point_cloud"].to(device)

                with torch.no_grad():
                    mean, log_variance = encoder(batch_val)
                    code = encoder.sample_latent_z(mean, log_variance)

                    timestep_batch = torch.randint(0, diffuser.T, (batch_val.shape[0],), device=device).long()
                    
                    noisy_pc, actual_noise = diffuser.add_noise(batch_val, timestep_batch)
                    predicted_noise = denoiser(noisy_pc, timestep_batch, code)

                    KLD_loss = encoder.KLD_loss(mean, log_variance)

                    mse_loss_per_item = F.mse_loss(predicted_noise, actual_noise, reduction='none').mean(dim=[1, 2]) #[B]
                    MSE_loss = mse_loss_per_item.mean()

                    loss = MSE_loss + 0.001 * KLD_loss 

                loss_total_val += loss.item()
            
            # Val Loss
            loss_val = loss_total_val / len(valloader)
            print(f'[{epoch:03d}] val_loss: {loss_val:.3f}')
            writer_val.add_scalar("Loss", loss_val, iteration*config['train_batch_size'])
            
            # Average loss per timestep     
            for t_val, item_mse in zip(timestep_batch.tolist(), mse_loss_per_item.tolist()):
                val_timestep_loss_sum[t_val] += item_mse
                val_timestep_counts[t_val] += 1

            avg_error_list = []
            valid_timesteps = []
            for time_val in range(diffuser.T):
                if timestep_counts[time_val] > 0:
                    avg_error = val_timestep_loss_sum[time_val] / val_timestep_counts[time_val]
                    avg_error_list.append(avg_error)
                    valid_timesteps.append(time_val)
                    writer_val.add_scalar(f"Val MSE Timestep Error/Epoch_{epoch:03d}", avg_error, global_step=time_val)

            # Chamfer
            #___________________________________________________________________________________________________________________
            number_of_points = 2048
            DDIM_steps = 50
            valset = valloader.dataset
            avg_chamfer = 0.
            with torch.no_grad():
                for object_index in range(valset.real_length):
                    pc = valset[object_index]['point_cloud'].to(device).unsqueeze(0)
                    mean, log_variance = encoder(pc) # valset not in this function
                    code = encoder.sample_latent_z(mean, log_variance)

                    generated_pc_ddim = sample_ddim(decoder, code, n_points=number_of_points, steps=DDIM_steps, timesteps=config['timesteps'])
                    
                    dist = chamfer_distance(pc, generated_pc_ddim).item()
                    avg_chamfer += dist
                    # print(f"Chamfer distance, Object{object_index}: {dist}")
            avg_chamfer_dist = avg_chamfer / (valset.real_length)
            print(f"[{epoch:03d}] avg_chamf: {avg_chamfer_dist:.05f}")
            writer_val.add_scalar("Average Chamfer Distance", avg_chamfer_dist, epoch)

            if avg_chamfer_dist < model_config['best_chamfer_loss']:
                model_config['best_chamfer_loss'] = avg_chamfer_dist
                utils.save_model(encoder, decoder, optimizer, scheduler, config, model_config, type = 'best_chamfer')
                # print(f"Best Chamfer Model:  {model_config['best_chamfer_loss']:.05f}")

            # Chamfer
            #___________________________________________________________________________________________________________________


            if loss_val < model_config['best_val_loss']:
                model_config['best_val_loss'] = loss_val
                utils.save_model(encoder, decoder, optimizer, scheduler, config, model_config, type = 'best_val')
                # print(f"Best val Model:  {model_config['best_val_loss']:.05f}")

            encoder.train()
            decoder.train()

            # Clear dictionaries so the next epoch tracks completely fresh averages
            val_timestep_loss_sum.clear()
            val_timestep_counts.clear()

            # TRAIN CHAMFER
            #_____________________________________________________________________________________________________________

            number_of_points = 2048
            DDIM_steps = 50
            trainset = trainloader.dataset
            avg_chamfer = 0.
            num_of_train_obj = 20
            with torch.no_grad():
                for object_index in random.sample(range(0, trainset.real_length - 1), num_of_train_obj):
                    pc = trainset[object_index]['point_cloud'].to(device).unsqueeze(0)
                    mean, log_variance = encoder(pc) # valset not in this function
                    code = encoder.sample_latent_z(mean, log_variance)

                    generated_pc_ddim = sample_ddim(decoder, code, n_points=number_of_points, steps=DDIM_steps, timesteps=config['timesteps'])
                    
                    dist = chamfer_distance(pc, generated_pc_ddim).item()
                    avg_chamfer += dist
                    # print(f"Chamfer distance, Object{object_index}: {dist}")
            avg_chamfer_dist = avg_chamfer / (num_of_train_obj)
            writer_train.add_scalar("Average Chamfer Distance", avg_chamfer_dist, epoch)

            # TRAIN CHAMFER
            #_____________________________________________________________________________________________________________
        
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
            utils.save_model(encoder, decoder, optimizer, scheduler, config, model_config, type = "best_train")
            print(f"Best train Model: {model_config['best_train_loss']:.03f}")
            #DEBUG
            # debug_file.write('best_train_loss: ' + str(best_train_loss) + '\n')
            # debug_file.flush()
            #DEBUG

        
        # Checkpoint saving
        model_config['last_epoch'] = epoch
        utils.save_model(encoder, decoder, optimizer, scheduler, config, model_config, type = 'checkpoint')


    print('Best Loss: ', model_config['best_train_loss'])
    print('Best Val Loss: ', model_config['best_val_loss'])
    print('Best Chamfer:  ', model_config['best_chamfer_loss'])
    print('Best Integral:  ', model_config['best_integral'])






if __name__ == "__main__":

    # Model name
    current_time = datetime.datetime.now().strftime("%b%d_%H-%M")
    denoiser_name = "7000epochs_128latent_enc128_dec512_globalnorm"
    experiment_name = f"{current_time}_{denoiser_name}"


    config = {
        'experiment_name': experiment_name,
        'device': 'cuda:0',
        'train_batch_size': 64,
        'val_batch_size': 128,
        'learning_rate': 0.0004,
        'step_size': 10, # scheduler step, one step is one batch
        'gamma': 1,
        'max_epochs': 7000,
        'timesteps': 1000,
        'print_every_n': 30, # every n batches
        'validate_every_n_epochs': 15,
    }

    model_config = {
        'last_epoch': 0,
        'enc_hidden_channels': 128,
        'dec_hidden_dim': 512,
        'latent_dim': 128,
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
    trainset = dataset.Dataset_new('train', config['timesteps'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['train_batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True)
    valset = dataset.Dataset_new('val', config['timesteps'])
    valloader = torch.utils.data.DataLoader(valset, batch_size=config['val_batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True)

    encoder = Encoder(
        number_points=2048, in_channels=3, 
        hidden_channels=model_config['enc_hidden_channels'], latent_dim=model_config['latent_dim'], clamp=False)
    decoder = Decoder(
        number_points=2048, point_dim=3, 
        hidden_dim=model_config['dec_hidden_dim'], latent_dim=model_config['latent_dim'], 
        timesteps=config['timesteps'], beta_start=1e-4, beta_end=0.02)

    encoder.to(device)
    decoder.to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['step_size'], config['gamma'])


    total_enc, trainable_enc = utils.count_parameters(encoder)
    total_dec, trainable_dec = utils.count_parameters(decoder)
    print(f"Encoder Total: {total_enc:,} | Trainable: {trainable_enc:,} | Model size: {utils.model_memory_size(encoder):.3f} MB")
    print(f"Decoder Total: {total_dec:,} | Trainable: {trainable_dec:,} | Model size: {utils.model_memory_size(decoder):.3f} MB")

    torch.cuda.empty_cache()
    #tensorboard --logdir=logs
    
    # #Reload model
    # experiment_name = "Jun16_17-40_100_Objects_1000epochs_64latent_enc128_dec256_globalnorm"
    # config, model_config, encoder, decoder = utils.reload_model(optimizer, scheduler, experiment_name, 'checkpoint', device)
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
            encoder, decoder, trainloader, valloader, device, optimizer, scheduler, config, model_config, writer_train, writer_val, debug_file)

    writer_train.close()
    writer_val.close()