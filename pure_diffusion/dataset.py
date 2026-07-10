import torch
from pathlib import Path
import numpy as np
import trimesh
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
    
class Dataset_grasp_and_pc(Dataset):
    """
    Dataset for loading normalized point clouds as .npz files into RAM.
    """
    # Adjust this path to match your actual output directory from the preprocessing script
    dataset_path = Path("/home/nikola/tum-adlr-ss26-07/diffusion_autoencoder/data") 

    def __init__(self, split):
        """
        :param split: one of 'train', 'val' or 'test' - for training, validation or test split
        :param timesteps: the artificial length of an epoch (useful for diffusion models)
        """
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.split = split
        
        split_dir = self.dataset_path / split
        file_paths = sorted(list(split_dir.glob("*.npz")), key=lambda x: x.name)        
        all_pcs = []
        all_grasps = []
        
        for fp in tqdm(file_paths, desc=f"Loading {split} data"):
            data = np.load(fp)
            
            # Arrays are shape: [num_grasps, 2048, 3] and [num_grasps, 12]
            all_pcs.append(data["point_clouds"])
            all_grasps.append(data["joint_angles"])
            
        # Concatenate everything into massive NumPy arrays
        # This effectively flattens the dataset so 1 index = 1 grasp
        if len(all_pcs) > 0:
            np_pcs = np.concatenate(all_pcs, axis=0)
            np_grasps = np.concatenate(all_grasps, axis=0)
        else:
            raise RuntimeError(f"No .npz files found in {split_dir}!")
            
        # Convert to compact PyTorch tensors
        # We keep these on the CPU RAM. The DataLoader workers will read from here
        # and move batches to the GPU (device) during the training loop.
        self.point_clouds = torch.tensor(np_pcs, dtype=torch.float32)
        self.grasps = torch.tensor(np_grasps, dtype=torch.float32)
        
    def __getitem__(self, index):
        """
        Retrieves a single grasp sample.
        """
        
        # Returning a dictionary makes it very clean to unpack in the training loop
        return {
            "point_cloud": self.point_clouds[index],
            "grasp": self.grasps[index],
        }
        
    def __len__(self):
        """
        :return: inflated length of the dataset dictated by timesteps
        """
        return len(self.grasps)