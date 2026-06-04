import os
import argparse
import torch
from pytorch3d.loss import chamfer_distance
import numpy as np
np.bool8 = np.bool
np.string_ = np.bytes_
np.unicode_ = np.str_
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from dataset import PointCloudDataset, get_point_cloud_files
from preprocessing import *
from common import *
from encoder import *
from decoder import *
from visualization import *
print("imports finished")

def check_path(path:str):
    if Path(path).exists():
        print("Path " + path + " exists")
    else:
        print(path + " does not exist")

def kl_annealing(model:AutoEncoder, current_epoch, num_epochs, kl_start, kl_end):
    fraction = (kl_end - kl_start) / num_epochs
    model.kl_state = kl_start + current_epoch * fraction

parser = argparse.ArgumentParser()

# model arguments
parser.add_argument('--latent_dim', type=int, default=32)
parser.add_argument('--num_steps', type=int, default=200)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--resume', type=str, default=None)
parser.add_argument("--num_points", type=int, default=256)
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--save_path", type=str, default="./models")
parser.add_argument("--kl_start", type=float, default=1e-4)
parser.add_argument("--kl_end", type=float, default=0.01)

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default="../data/preprocessed/preprocessing1")
parser.add_argument('--train_batch_size', type=int, default=1)
parser.add_argument('--val_batch_size', type=int, default=1)
parser.add_argument('--rotate', type=eval, default=False, choices=[True, False])

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
parser.add_argument("--num_epochs", type=int, default=50)
args = parser.parse_args()
print("Arguments parsed")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Declare device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device: ", device)

# Checking if Paths are correct
print("Checking Paths")
check_path(args.dataset_path)
check_path(args.save_path)
if args.resume:
    check_path(args.resume)

# Logging
if args.logging:
    writer = torch.utils.tensorboard.SummaryWriter(args.log_root)


# Dataset and Dataloader
point_cloud_files = get_point_cloud_files(args.dataset_path)
dataset_size = len(point_cloud_files)
# Train, Val, Test split hardcoded at the moment
train_val_test_split = {
    "train_size": int(dataset_size * 0.8),
    "val_size": int(dataset_size * 0.1),
    "test_size": int(dataset_size - (int(dataset_size * 0.8) + int(dataset_size * 0.1)))
}

train_dataset = PointCloudDataset(point_cloud_files[:train_val_test_split["train_size"]], number_points=args.num_points)
val_dataset = PointCloudDataset(point_cloud_files[train_val_test_split["train_size"]:(train_val_test_split["val_size"] + train_val_test_split["train_size"])],
                                number_points=args.num_points)
test_dataset = PointCloudDataset(point_cloud_files[-train_val_test_split["test_size"]:], number_points=args.num_points)


print(f"Dataset Size: {dataset_size}")
print(f"Training Size {len(train_dataset)}\t|\t Val Size {len(val_dataset)}\t|\tTest Size {len(test_dataset)}")

train_dl = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=args.val_batch_size, drop_last=True)
test_dl = DataLoader(test_dataset, batch_size=args.train_batch_size)

print(f"\nTraining_DL Size {len(train_dl)}\t|\t Val_DL Size {len(val_dl)}\t|\tTest_DL Size {len(test_dl)}")

if args.resume:
    # Resuming Training from checkpoint
    model_state_dict = torch.load(args.resume)
    model = AutoEncoder(model_state_dict["args"]).to(device)
    model.load_state_dict(model_state_dict["state_dict"])
else:
    model = AutoEncoder(args.num_points, 3, args.hidden_dim, args.latent_dim, args.num_steps, args.beta_1, args.beta_T, args.kl_start)

    print(f"\nBuilding new Model with num_points: {args.num_points}, hidden_dim: {args.hidden_dim}, latent_dim: {args.latent_dim} \
          \nnum_steps: {args.num_steps}, beta_1: {args.beta_1}, beta_T: {args.beta_T}, kl_start: {args.kl_start}, kl_end: {args.kl_end}")

num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nNumber of Trainable Parameters: {num_parameters / 1e6:.2f}M")

# Optimizer and LR_Scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)

# Train, validate 
def train(train_batch, val_batch):
    # Load data
    train_batch.to(device)
    val_batch.to(device)
    # Reset grad and model state
    optimizer.zero_grad()

    # Forward
    loss, kld_loss = model.get_loss(train_batch)
    val_loss, _ = model.get_loss(val_batch)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()

    return loss.item(), val_loss.item(), kld_loss.item(), orig_grad_norm
    

if args.dry_run == False:
    patience = 0
    previous_val_loss = 1e6
    print(f"Starting training with {args.num_epochs} epochs")
    train_loss_history = []
    val_loss_history = []
    kld_loss_history = []

    for i in range(args.num_epochs):
        epoch_train_loss = 0
        epoch_val_loss = 0
        epoch_kld_loss = 0
        kl_annealing(model, i+1, args.num_epochs, args.kl_start, args.kl_end)

        for it, train_batch in enumerate(train_dl):
            val_batch = next(iter(val_dl))
            train_loss, val_loss, kld_loss, orig_grad_norm = train(train_batch, val_batch)

            epoch_train_loss += train_loss
            epoch_val_loss += val_loss
            epoch_kld_loss += kld_loss
        
        if epoch_val_loss > previous_val_loss:
            patience += 1
        if patience >= args.patience:
            scheduler.step()
            patience = 0

        if epoch_val_loss < previous_val_loss:
            torch.save(model.state_dict(), args.save_path + "/best_model.pt")

        previous_val_loss = epoch_val_loss


        writer.add_scalar('train/loss', epoch_train_loss, i)
        writer.add_scalar("val/loss", epoch_val_loss, i)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], i)
        writer.add_scalar('train/grad_norm', orig_grad_norm, i)
        #writer.flush()
        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss)
        kld_loss_history.append(epoch_kld_loss)

        print(f"Epoch: {i+1}\tTraining Loss: {epoch_train_loss:.3f}\tVal Loss: {epoch_val_loss:.3f}\tKLD Loss: {epoch_kld_loss:.3f}")

        

    torch.save(model.state_dict(), args.save_path + "/end_model.pt")
    plt.plot(train_loss_history, color="b", label="Train Loss")
    plt.plot(val_loss_history, color="r", label="Val Loss")
    plt.plot(kld_loss_history, color="m", label="KLD Loss")
    plt.legend()
    plt.show()




    
