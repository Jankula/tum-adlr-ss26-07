import torch
from pathlib import Path
import numpy as np
import trimesh
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
    

# class Dataset(Dataset):
#     """
#     Dataset for loading ShapeNet Voxels from disk
#     """
#     dataset_path = Path("data") 

#     def __init__(self, split, timesteps):
#         """
#         :param split: one of 'train', 'val' or 'overfit' - for training, validation or overfitting split
#         """
#         super().__init__()
#         assert split in ['train', 'val', 'overfit']
#         self.timesteps = timesteps
#         self.split = split
#         self.item_names = Path(f"data/splits/{split}.txt").read_text().splitlines()
#         self.real_length = len(self.item_names)

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       

#         temp_items = []
#         for item_name in self.item_names:
#             pcd_np = self.get_shape_pointcloud(Path(f"data/{item_name}"))
#             temp_items.append(pcd_np)
            
#         self.items = torch.tensor(np.array(temp_items), dtype=torch.float32).to(device)

#     def __getitem__(self, index):
#         return self.items[index % len(self.items)]
        

#     def __len__(self):
#         """
#         :return: length of the dataset
#         """
#         return self.timesteps


#     @staticmethod
#     def get_shape_pointcloud(path):
#         """
#         Utility method for reading a point cloud
#         :return: a numpy array representing the shape point cloud
#         """
#         pcd = trimesh.load(path, file_type = 'obj', force='pointcloud')
#         points = np.array(pcd.vertices)
        
#         return points
    

class Dataset_new(Dataset):
    """
    Dataset for loading normalized point clouds as .npz files into RAM.
    """
    # Adjust this path to match your actual output directory from the preprocessing script
    dataset_path = Path("data") 

    def __init__(self, split, timesteps):
        """
        :param split: one of 'train', 'val' or 'test' - for training, validation or test split
        :param timesteps: the artificial length of an epoch (useful for diffusion models)
        """
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.timesteps = timesteps
        self.split = split
        
        split_dir = self.dataset_path / split
        file_paths = list(split_dir.glob("*.npz"))
        
        all_pcs = []
        # all_joints = []
        # all_scores = []
        
        for fp in tqdm(file_paths, desc=f"Loading {split} data"):
            data = np.load(fp)
            
            # Arrays are shape: [num_grasps, 2048, 3] and [num_grasps, 12]
            all_pcs.append(data["point_clouds"])
            # all_joints.append(data["joint_angles"])
            # all_scores.append(data["scores"])
            
        # 2. Concatenate everything into massive NumPy arrays
        # This effectively flattens the dataset so 1 index = 1 grasp
        if len(all_pcs) > 0:
            np_pcs = np.concatenate(all_pcs, axis=0)
            # np_joints = np.concatenate(all_joints, axis=0)
            # np_scores = np.concatenate(all_scores, axis=0)
        else:
            raise RuntimeError(f"No .npz files found in {split_dir}!")
            
        # 3. Convert to compact PyTorch tensors
        # We keep these on the CPU RAM. The DataLoader workers will read from here
        # and move batches to the GPU (device) during the training loop.
        self.point_clouds = torch.tensor(np_pcs, dtype=torch.float32)
        # self.joint_angles = torch.tensor(np_joints, dtype=torch.float32)
        # self.scores = torch.tensor(np_scores, dtype=torch.float32)
        
        self.real_length = len(self.point_clouds)

    def __getitem__(self, index):
        """
        Retrieves a single grasp sample.
        Because timesteps > real_length, we use modulo to loop over the real data.
        """
        real_idx = index % self.real_length
        
        # Returning a dictionary makes it very clean to unpack in the training loop
        return {
            "point_cloud": self.point_clouds[real_idx],
            # "joint_angles": self.joint_angles[real_idx],
            # "score": self.scores[real_idx]
        }
        
    def __len__(self):
        """
        :return: inflated length of the dataset dictated by timesteps
        """
        return self.timesteps