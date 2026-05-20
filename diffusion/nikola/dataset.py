import torch
from pathlib import Path
import json
import numpy as np
import trimesh

# class Data():
#     def __init__(self, path):
#         self.path = path
#         self.points = self.load_data()
        
#     def load_data(self):
#         pcd = trimesh.load(self.path, file_type = 'obj', force='pointcloud')
#         points = np.array(pcd.vertices)
        
#         return points
    

class Dataset(torch.utils.data.Dataset):
    """
    Dataset for loading ShapeNet Voxels from disk
    """
    dataset_path = Path("data") 

    def __init__(self, split, timesteps):
        """
        :param split: one of 'train', 'val' or 'overfit' - for training, validation or overfitting split
        """
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        self.timesteps = timesteps
        self.split = split
        self.items = Path(f"data/splits/{split}.txt").read_text().splitlines()

    def __getitem__(self, index):

        item = self.items[index % len(self.items)]
        # pcd = self.get_shape_pointcloud(Path(f"data/{item}"))
        # return pcd[np.newaxis, :, :]  # [B, N, 3]
        pcd_np = self.get_shape_pointcloud(Path(f"data/{item}"))
        pcd_tensor = torch.tensor(pcd_np, dtype=torch.float32)
        return pcd_tensor
        

    def __len__(self):
        """
        :return: length of the dataset
        """
        return len(self.items) if self.split != 'overfit' else self.timesteps


    @staticmethod
    def get_shape_pointcloud(path):
        """
        Utility method for reading a point cloud
        :return: a numpy array representing the shape point cloud
        """
        pcd = trimesh.load(path, file_type = 'obj', force='pointcloud')
        points = np.array(pcd.vertices)
        
        return points