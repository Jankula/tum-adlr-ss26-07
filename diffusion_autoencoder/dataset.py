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
            
        self.items = torch.tensor(np.array(temp_items), dtype=torch.float32).to(device)

    def __getitem__(self, index):
        return self.items[index % len(self.items)]
        

    def __len__(self):
        """
        :return: length of the dataset
        """
        return self.timesteps


    @staticmethod
    def get_shape_pointcloud(path):
        """
        Utility method for reading a point cloud
        :return: a numpy array representing the shape point cloud
        """
        pcd = trimesh.load(path, file_type = 'obj', force='pointcloud')
        points = np.array(pcd.vertices)
        
        return points

def get_point_cloud_files(root:Path):
    if not isinstance(root, Path):
        root = Path(root)
    if not root.exists:
        print("Path does not exist")
        return []
    point_cloud_files = sorted(root.rglob("pointcloud*.obj"))
    return point_cloud_files


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, files:list, max_objects=None, number_points=2048):
        super().__init__()

        self.files = files
        if max_objects:
            self.files = self.files[:max_objects]
            
        self.number_points = number_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        point_cloud_path = self.files[index]

        point_cloud = np.array(
            trimesh.load(point_cloud_path).vertices,
            dtype=np.float32
        )[:self.number_points]

        return torch.from_numpy(point_cloud)