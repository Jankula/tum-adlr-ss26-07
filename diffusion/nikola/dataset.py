import torch
from pathlib import Path
import json
import numpy as np
import trimesh
import torch

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
        self.item_names = Path(f"data/splits/{split}.txt").read_text().splitlines()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       

        temp_items = []
        for item_name in self.item_names:
            pcd_np = self.get_shape_pointcloud(Path(f"data/{item_name}"))
            temp_items.append(pcd_np)
            # pcd_tensor = torch.tensor(pcd_np, dtype=torch.float32)
            # if torch.cuda.is_available():
            #     pcd_tensor = pcd_tensor.to(device)
            # self.items.append(pcd_tensor)
            
        self.items = torch.tensor(np.array(temp_items), dtype=torch.float32).to(device)

    def __getitem__(self, index):
        # item = self.items[index % len(self.items)]
        # pcd_np = self.get_shape_pointcloud(Path(f"data/{item}"))
        # pcd_tensor = torch.tensor(pcd_np, dtype=torch.float32)
        # return pcd_tensor

        return self.items[index % len(self.items)]
        

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