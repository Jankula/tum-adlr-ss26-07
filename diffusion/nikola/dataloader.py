import numpy as np
import trimesh

class Data:
    def __init__(self, path):
        self.path = path
        self.points = self.load_data()
        
    def load_data(self):
        pcd = trimesh.load(self.path, file_type = 'obj', force='pointcloud')
        points = np.array(pcd.vertices)
        
        return points
