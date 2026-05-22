import os
import math
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import trimesh
import torch


def get_point_cloud_files():
    root = Path.cwd()
    data_dir = root / Path("data/preprocessed/preprocessing2")
    point_cloud_files = sorted(data_dir.rglob("pointcloud*.obj"))
    return point_cloud_files

class PointCloudDataset(Dataset):
    def __init__(self, max_objects=None, number_points=2048):
        super().__init__()
        self.files = get_point_cloud_files()[:max_objects]
        self.number_points = number_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        point_cloud = np.array(trimesh.load(self.files[index]).vertices, dtype="float32")[:self.number_points]
        return torch.tensor(point_cloud)
