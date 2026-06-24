import os
import argparse
import torch
#from pytorch3d.loss import chamfer_distance
import numpy as np
np.bool8 = np.bool
np.string_ = np.bytes_
np.unicode_ = np.str_
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from time import strftime, localtime

from dataset import PointCloudDataset, get_point_cloud_files
from preprocessing import *
from common import *
from encoder import *
from decoder import *
from visualization import *

class Logger:
    def __init__(self,):
        self.line = []
    
    def __call__(self, line:str):
        self.line.append(line)
    
    def write_log_file(self, save_path):
        with open(save_path + "/train_log.txt", 'w') as f:
            for line in self.line:
                print(line, file=f)

def log_args(logger:Logger, args:argparse.ArgumentParser):
    logger("Argparser Arguments:")
    logger("Model arguments:")
    logger(f"latent_dim: {args.latent_dim}")
    logger(f"num_steps: {args.num_steps}")
    logger(f"beta_1: {args.beta_1}")
    logger(f"beta_T: {args.beta_T}")
    logger(f"sched_mode: {args.sched_mode}")
    logger(f"flexibility: {args.flexibility}")
    if args.resume:
        logger("Continuing Training of model: " + args.resume)
    else:
        logger("Training with new model:")
    logger(f"num_points: {args.num_points}")
    logger(f"hidden_dim: {args.hidden_dim}")
    logger(f"save_path: {args.save_path}")
    logger(f"kl_start: {args.kl_start}")
    logger(f"kl_end: {args.kl_end}")
    logger("Datset and Dataloader Arguments:")
    logger(f"datset_path: {args.dataset_path}")
    logger(f"train_batch_size: {args.train_batch_size}")
    logger(f"val_batch_size: {args.val_batch_size}")
    logger("Arguments of Optimizer and Scheduler:")
    logger(f"lr: {args.lr}")
    logger(f"weight_decay: {args.weight_decay}")
    logger(f"max_grad_norm: {args.max_grad_norm}")
    logger(f"end_lr: {args.end_lr}")
    logger(f"sched_start_epoch: {args.sched_start_epoch}")
    logger(f"sched_end_epoch: {args.sched_end_epoch}")
    logger("Training Arguments:")
    logger(f"seed: {args.seed}")
    logger(f"logging: {args.logging}")
    logger(f"log_root: {args.log_root}")
    logger(f"dry_run: {args.dry_run}")
    logger(f"patience: {args.patience}")
    logger(f"num_epochs: {args.num_epochs}")


def check_path(path:str):
    if Path(path).exists():
        return "Path " + path + " exists"
    else:
        return path + " does not exist"

def kl_annealing(model:AutoEncoder, current_epoch, num_epochs, kl_start, kl_end):
    fraction = (kl_end - kl_start) / num_epochs
    model.kl_state = kl_start + current_epoch * fraction

def chamfer_custom(pc_a, pc_b):
    # pc_a: [B, N, 3]
    # pc_b: [B, M, 3]

    dists = torch.cdist(pc_a, pc_b, p=2) ** 2
    # shape: [B, N, M]

    pc_a_to_pc_b = dists.min(dim=2).values  # [B, N]
    pc_b_to_pc_a = dists.min(dim=1).values  # [B, M]

    return pc_a_to_pc_b.mean() + pc_b_to_pc_a.mean()

parser = argparse.ArgumentParser()

# model arguments
parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--num_steps', type=int, default=200)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument("--num_points", type=int, default=512)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--save_path", type=str, default="./models")
parser.add_argument("--kl_start", type=float, default=1e-4)
parser.add_argument("--kl_end", type=float, default=0.01)

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default="../data/preprocessed/preprocessing5")
parser.add_argument('--test_path', type=str, default="../data/preprocessed/test")
parser.add_argument('--train_batch_size', type=int, default=3)
parser.add_argument('--val_batch_size', type=int, default=3)

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=5)
parser.add_argument('--sched_end_epoch', type=int, default=10)

# Training
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_ae')
parser.add_argument("--dry_run", default=False, type=bool, choices=[True, False])
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--num_epochs", type=int, default=10)
args = parser.parse_args()
print("Arguments parsed")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

#declare logger
logger = Logger()
logger(f"AutoEncoder(number_points={args.num_points}, point_dim=3, hidden_dim={args.hidden_dim}, latent_dim={args.latent_dim}, num_steps={args.num_steps},\
beta_1={args.beta_1}, beta_T={args.beta_T}, kl_start={args.kl_start})")
log_args(logger, args)

# Declare device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device: ", device)
logger("Using device: " + str(device) + '\n')

# Checking if Paths are correct
print("Checking Paths")
logger("Checking Paths\n")

print(check_path(args.dataset_path))
logger(check_path(args.dataset_path))

print(check_path(args.save_path) + '\n')
logger(check_path(args.save_path) + '\n')

# creating save path
save_path = args.save_path + strftime("/%b_%a_%H_%M_%S", localtime())
print("Creating save Path: " + save_path)
logger("Creating save Path: " + save_path + '\n')

if not os.path.exists(save_path):
    os.mkdir(save_path)
if args.resume:
    print(check_path(args.resume))
    logger(check_path(args.resume) + '\n')

# Logging
log_path = save_path + args.log_root
print("creating log path: " + log_path)
logger("creating log path: " + log_path + '\n')
if not os.path.exists(log_path):
    os.mkdir(log_path)
if args.logging:
    writer = torch.utils.tensorboard.SummaryWriter(log_path)


# Dataset and Dataloader
point_cloud_files = get_point_cloud_files(args.dataset_path)
test_files = get_point_cloud_files(args.test_path)
dataset_size = len(point_cloud_files)
logger("Size of the whole Training Dataset: " + str(dataset_size) + '\n')
# Train, Val, Test split hardcoded at the moment
train_val_test_split = {
    "train_size": int(dataset_size * 0.9),
    "val_size": int(dataset_size * 0.1),
}

train_dataset = PointCloudDataset(point_cloud_files[:train_val_test_split["train_size"]], number_points=args.num_points)
val_dataset = PointCloudDataset(point_cloud_files[train_val_test_split["train_size"]:],
                                number_points=args.num_points)
test_dataset = PointCloudDataset(test_files, number_points=args.num_points)


print(f"Dataset Size: {dataset_size}")
print(f"Training Size {len(train_dataset)}\t|\t Val Size {len(val_dataset)}\t|\tTest Size {len(test_dataset)}")
logger("Dataset Sizes after splitting:\n" + f"Training Size {len(train_dataset)}\t|\t Val Size {len(val_dataset)}\t|\tTest Size {len(test_dataset)}\n")

train_dl = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
val_dl = DataLoader(val_dataset, batch_size=args.val_batch_size, drop_last=True)
test_dl = DataLoader(test_dataset, batch_size=50, drop_last=True)

print(f"\nTraining_DL Size {len(train_dl)}\t|\t Val_DL Size {len(val_dl)}\t|\tTest_DL Size {len(test_dl)}")
logger("Dataloader Sizes after Splitting:" + f"\nTraining_DL Size {len(train_dl)}\t|\t Val_DL Size {len(val_dl)}\t|\tTest_DL Size {len(test_dl)}\n")

if args.resume:
    # Resuming Training from checkpoint
    model_state_dict = torch.load(args.resume)
    model = AutoEncoder(model_state_dict["args"]).to(device)
    model.load_state_dict(model_state_dict["state_dict"])
else:
    model = AutoEncoder(args.num_points, 3, args.hidden_dim, args.latent_dim, args.num_steps, args.beta_1, args.beta_T, args.kl_start).to(device)

    print(f"\nBuilding new Model with num_points: {args.num_points}, hidden_dim: {args.hidden_dim}, latent_dim: {args.latent_dim} \
          \nnum_steps: {args.num_steps}, beta_1: {args.beta_1}, beta_T: {args.beta_T}, kl_start: {args.kl_start}, kl_end: {args.kl_end}")
    
    logger(f"\nBuilding new Model with num_points: {args.num_points}, hidden_dim: {args.hidden_dim}, latent_dim: {args.latent_dim} \
          \nnum_steps: {args.num_steps}, beta_1: {args.beta_1}, beta_T: {args.beta_T}, kl_start: {args.kl_start}, kl_end: {args.kl_end}\n")

num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nNumber of Trainable Parameters: {num_parameters / 1e6:.2f}M")
logger(f"\nNumber of Trainable Parameters: {num_parameters / 1e6:.2f}M\n")

# Optimizer and LR_Scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
logger(f"Using Starting Learning Rate: {args.lr} and Weight Decay: {args.weight_decay}\n")

scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)
logger("Scheduler Parameters:\n" + f"Start Epoch: {args.sched_start_epoch}\n" \
       + f"End Epoch: {args.sched_end_epoch}\n" + f"Learning Rate: {args.lr}" + f"End Learning Rate: {args.end_lr}\n")

print("Training on: ", next(model.parameters()).device)
logger(f"Training on: {next(model.parameters()).device}")


# Train and model updates
def train(train_batch):
    # Load data
    train_batch = train_batch.to(device)
    # Reset grad and model state
    optimizer.zero_grad()

    # Forward
    loss, kld_loss = model.get_loss(train_batch)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()

    return loss.item(), kld_loss.item(), orig_grad_norm


@torch.no_grad
def validate(val_batch):
    val_batch = val_batch.to(device)
    val_loss, _ = model.get_loss(val_batch)
    return val_loss.item()


# Train the model if dry_run == false
if args.dry_run == False:
    patience = 0
    previous_chamfer_loss = 1e6
    previous_val_loss = 1e6
    print(f"Starting training with {args.num_epochs} epochs")
    logger(f"Starting training with {args.num_epochs} epochs\n")
    train_loss_history = []
    val_loss_history = []
    kld_loss_history = []
    chamfer_loss_history = []
    epoch = []

    for i in range(args.num_epochs):
        epoch_train_loss = 0
        epoch_val_loss = 0
        epoch_kld_loss = 0
        kl_annealing(model, i+1, args.num_epochs, args.kl_start, args.kl_end)

        for it, train_batch in enumerate(train_dl):
            train_loss, kld_loss, orig_grad_norm = train(train_batch)

            epoch_train_loss += train_loss
            epoch_kld_loss += kld_loss
        
        epoch_train_loss /= len(train_dl)
        epoch_kld_loss /= len(train_dl)
        
        model.eval()
        
        for val_batch in val_dl:
            val_loss = validate(val_batch)
            epoch_val_loss += val_loss
        
        epoch_val_loss /= len(val_dl)
        
        if epoch_val_loss > previous_val_loss:
            patience += 1
        
        if patience >= args.patience:
            scheduler.step()
            logger(f"Reducing Learning Rate at Epoch: {i+1}\n" + f"New Learning Rate: {optimizer.param_groups[0]['lr']:.5f}\n")
            patience = 0
            
        model.train()
        
        if(i+1) % 50 == 0:
            
            model.eval()
            
            chamfer_loss = 0
            
            for batch in val_dl:
                _, means, _ = model.encode(batch)
                pred_pcs = model.decode(means, args.num_points)
                chamfer_loss += chamfer_custom(pred_pcs, batch)
            
            if bool(chamfer_loss.shape):
                chamfer_loss = chamfer_loss.mean()

            if chamfer_loss < previous_chamfer_loss:
                logger(f"Saving new best model at Epoch: {i+1}\n" + f"Chamfer loss of the best model: {chamfer_loss:.3f}\n")
                torch.save(model.state_dict(), save_path + "/best_model.pt")

            chamfer_loss_history.append(chamfer_loss)
            epoch.append((i+1))
            previous_chamfer_loss = chamfer_loss
            
            model.train()


        writer.add_scalar('train/loss', epoch_train_loss, i)
        writer.add_scalar("val/loss", epoch_val_loss, i)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], i)
        writer.add_scalar('train/grad_norm', orig_grad_norm, i)
        #writer.flush()
        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss)
        kld_loss_history.append(epoch_kld_loss)

        if (i+1) % 1 == 0:
            print(f"Epoch: {i+1}\tTraining Loss: {epoch_train_loss:.3f}\tVal Loss: {epoch_val_loss:.3f}\tKLD Loss: {epoch_kld_loss:.3f}")
            logger(f"Epoch: {i+1}\tTraining Loss: {epoch_train_loss:.3f}\tVal Loss: {epoch_val_loss:.3f}\tKLD Loss: {epoch_kld_loss:.3f}")

    # calculating the accuracy of the model with the test dataset
    running_test_accuracy_means = 0
    model.eval()
    print("Start Testing")
    logger("Start Testing\n")
    for batch in test_dl:
        latents, mean, _ = model.encode(batch)
        pred_pc_means = model.decode(mean, args.num_points)
        chamfer_means = chamfer_custom(batch, pred_pc_means)
        if bool(chamfer_means.shape):
            chamfer_means = chamfer_means.mean(dim=0)
        
        running_test_accuracy_means += chamfer_means.detach().cpu().numpy()
    
    running_test_accuracy_means /= len(test_dl)
    
    logger(f"Test Accuracy with means: {running_test_accuracy_means / len(test_dl):.3f}\n")
    print(f"Test Accuracy: {running_test_accuracy_means:.3f}") 

    torch.save(model.state_dict(), save_path + "/end_model.pt")
    logger("Saving model at end of training: " + save_path + "/end_model.pt")
    logger(f"Val loss of model at end of training: {val_loss_history[-1]:.3f}")
    
    logger("Saving Plot: " + save_path + "/loss_history.png")
    logger.write_log_file(save_path)
    fig = plt.figure(figsize=(15, 10))
    plt.plot(train_loss_history, color="b", label="Train Loss")
    plt.plot(val_loss_history, color="r", label="Val Loss")
    plt.legend()
    fig.savefig(save_path + "/loss_history.png")
    fig = plt.figure(figsize=(15,10))
    plt.plot(kld_loss_history, color="m", label="KLD Loss")
    plt.legend()
    fig.savefig(save_path + "/kld_loss.png")
    
    fig = plt.figure(figsize=(15, 10))
    plt.plot(epoch, chamfer_loss_history, color="g", label="Chamfer Loss")
    plt.legend()
    fig.savefig(save_path + "/chamfer_loss.png")
    plt.show()
 




    
