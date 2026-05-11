import os
import math
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import trimesh


def get_point_cloud_files():
    root = Path.cwd()
    data_dir = root / Path("data/preprocessed/")
    point_cloud_files = sorted(data_dir.rglob("**/pointcloud*.obj"))
    return point_cloud_files


class PointCloudDataset(Dataset):
    def __init__(self, max_objects=None):
        super().__init__()
        self.files = get_point_cloud_files()[:max_objects]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        point_cloud = np.array(trimesh.load(self.files[index]).vertices, dtype="float32")
        return self.transform(point_cloud)
    