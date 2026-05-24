import pathlib
import os
import shutil
from collections import defaultdict
import datetime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

import network, utils, dataset, metrics, visualization


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


def train(denoiser, diffuser, trainloader, valloader, device, optimizer, scheduler, config, model_config, writer, debug_file):

    optimizer = optimizer
    # scheduler = scheduler

    denoiser.train()

    best_train_loss = 10
    train_loss_running = 0.
    # MSE_loss_running = 0.
    # SNR_loss_running = 0.
    # Chamfer_loss_running = 0.
    # Damped_Chamfer_loss_running = 0.

    # For tracking loss per timestep
    timestep_loss_sum = defaultdict(float)
    timestep_counts = defaultdict(int)

    # Config parameters printed in tensorboard
    config_markdown = "| Hyperparameter | Value |\n| :--- | :--- |\n"
    for key, value in config.items():
        config_markdown += f"| **{key}** | {value} |\n"
    writer.add_text("Hyperparameters", config_markdown, global_step=0)

    for epoch in range(model_config['last_epoch'], model_config['last_epoch'] + config['max_epochs']):
        
        train_loss_epoch_running = 0
        
        for i, batch in enumerate(trainloader):
            # print("Batch shape: " + str(batch.shape))#DEBUG
            
            batch = batch.to(device)

            optimizer.zero_grad()
            
            timestep_batch = torch.randint(0, diffuser.t, (batch.shape[0],), device=device).long()

            noisy_pc, actual_noise = diffuser.add_noise(batch, timestep_batch)
            predicted_noise = denoiser(noisy_pc, timestep_batch)

            # Loss calculation
            mse_loss_per_item = F.mse_loss(predicted_noise, actual_noise, reduction='none').mean(dim=[1, 2]) #[B]
            MSE_loss = mse_loss_per_item.mean()
            # SNR_loss = metrics.SNR(mse_loss_per_item, diffuser.alpha_cumprod, timestep_batch, gamma = 5.0)
            # Damped_Chamfer_loss, Chamfer_loss, predicted_x0 = metrics.compute_damped_geometry_loss(
                                                                    #     noisy_pc=noisy_pc,
                                                                    #     predicted_noise=predicted_noise,
                                                                    #     target_pc=batch,
                                                                    #     alpha_cumprod=diffuser.alpha_cumprod,
                                                                    #     timestep=timestep_batch,
                                                                    #     max_timesteps=diffuser.t
                                                                    # )
            # loss = SNR_loss + config['lambda'] * Damped_Chamfer_loss
            loss = MSE_loss

            loss.backward()
            optimizer.step()
            # scheduler.step()

            # print(f"Allocated pool: {torch.cuda.memory_reserved() / 1024**2:.2f} MiB") #DEBUG
            # print(f"True tensor usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB") #DEBUG

            # Loss logging
            train_loss_epoch_running += loss.item()
            train_loss_running += loss.item()
            # MSE_loss_running += MSE_loss.item()
            # SNR_loss_running += SNR_loss.item()
            # Chamfer_loss_running += Chamfer_loss.item()
            # Damped_Chamfer_loss_running += Damped_Chamfer_loss.item()

            #DEBUG
            debug_file.write('train_loss_epoch_running: ' + str(train_loss_epoch_running) + '\n')
            debug_file.flush()

            iteration = epoch * len(trainloader) + i

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                
                print(f'[{epoch:02d}/{i:03d}] train_loss: {train_loss_running / config["print_every_n"]:.3f}')
                writer.add_scalar("Train Loss", train_loss_running / config["print_every_n"], iteration)
                train_loss_running = 0.

                # # print(f'[{epoch:03d}/{i:05d}] MSE_loss: {MSE_loss_running / config["print_every_n"]:.3f}')
                # writer.add_scalar("MSE Loss", MSE_loss_running / config["print_every_n"], iteration)
                # MSE_loss_running = 0.

                # # print(f'[{epoch:03d}/{i:05d}] SNR_loss: {SNR_loss_running / config["print_every_n"]:.3f}')
                # writer.add_scalar("SNR Loss", SNR_loss_running / config["print_every_n"], iteration)
                # SNR_loss_running = 0.

                # # print(f'[{epoch:03d}/{i:05d}] Chamfer_loss: {Chamfer_loss_running / config["print_every_n"]:.3f}')
                # writer.add_scalar("Chamfer Loss", Chamfer_loss_running / config["print_every_n"], iteration)
                # Chamfer_loss_running = 0.

                # # print(f'[{epoch:03d}/{i:05d}] Damped_Chamfer_loss: {Damped_Chamfer_loss_running / config["print_every_n"]:.3f}')
                # writer.add_scalar("Damped Chamfer Loss", Damped_Chamfer_loss_running / config["print_every_n"], iteration)
                # Damped_Chamfer_loss_running = 0.


            # Average loss per timestep     
            for t_val, item_mse in zip(timestep_batch.tolist(), mse_loss_per_item.tolist()):
                timestep_loss_sum[t_val] += item_mse
                timestep_counts[t_val] += 1


            {
            # if iteration % config['print_EMD_every_n_batches'] == (config['print_EMD_every_n_batches'] - 1):
            #     EMD_loss = earth_mover_distance(predicted_x0[0:1].detach(), batch[0:1].detach()).mean()
            #     print(f'[{epoch:03d}/{i:05d}] EMD_loss: {EMD_loss:.3f}')

            # validation evaluation and logging
            # if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):

            #     # set model to eval, important if your network has e.g. dropout or batchnorm layers
            #     model.eval()

            #     loss_total_val = 0
            #     total, correct = 0, 0
            #     # forward pass and evaluation for entire validation set
            #     for batch_val in valloader:
            #         ShapeNetVox.move_batch_to_device(batch_val, device)
                    
            #         with torch.no_grad():
            #             # TODO: Get prediction scores
            #             prediction_val = model(batch_val['voxel']) #[B,9,NUM_CLASSES]

            #         # TODO: Get predicted labels from scores
            #         # print("Prediction val shape: " + str(prediction_val[:, 0, :].shape)) #DEBUG
            #         predicted_label = torch.argmax(prediction_val[:, 0, :], dim=1) #[B]

            #         # TODO: keep track of total / correct / loss_total_val
            #         for output_idx in range(prediction_val.shape[1]):
            #             loss_total_val += loss_criterion(prediction_val[:, output_idx, :], batch_val['label']).item()

            #         total += batch_val['label'].size(0)
            #         correct += (predicted_label == batch_val['label']).sum().item()

            #     accuracy = 100 * correct / total

            #     print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_total_val / len(valloader):.3f}, val_accuracy: {accuracy:.3f}%')

            #     if accuracy > best_accuracy:
            #         torch.save(model.state_dict(), f'exercise_2/runs/{config["experiment_name"]}/model_best.ckpt')
            #         best_accuracy = accuracy

            #     # set model back to train
            #     model.train()
            }

        # Average loss per timestep            
        for time_val in range(diffuser.t):
            if timestep_counts[time_val] > 0:
                avg_error = timestep_loss_sum[time_val] / timestep_counts[time_val]
                writer.add_scalar(f"MSE Timestep Error/Epoch_{epoch:03d}", avg_error, global_step=time_val)
                
        # Clear dictionaries so the next epoch tracks completely fresh averages
        timestep_loss_sum.clear()
        timestep_counts.clear()

        # Weight plotting
        for name, weight in denoiser.named_parameters():
            writer.add_histogram(f"Weights/{name}", weight, epoch)
            if weight.grad is not None:
              writer.add_histogram(f"Gradients/{name}", weight.grad, epoch)


        train_loss_epoch = train_loss_epoch_running / len(trainloader)
        #DEBUG
        debug_file.write('train_loss_epoch: ' + str(train_loss_epoch) + '\n')
        debug_file.flush()
        #DEBUG
        # Save model weights if the best loss is achieved  
        if  train_loss_epoch < best_train_loss:
            utils.save_model(denoiser, diffuser, optimizer, scheduler, config, model_config, type = "best")
            best_train_loss = train_loss_epoch
            print('Best Model:  ', best_train_loss)
            #DEBUG
            debug_file.write('best_train_loss: ' + str(best_train_loss) + '\n')
            debug_file.flush()
            #DEBUG

        
        # Model epoch saving
        model_config['last_epoch'] = epoch
        utils.save_model(denoiser, diffuser, optimizer, scheduler, config, model_config, type = 'epoch' + str(epoch))

        visualization.plot_interactive_epochs(config)



    print('Best Loss: ', best_train_loss)
