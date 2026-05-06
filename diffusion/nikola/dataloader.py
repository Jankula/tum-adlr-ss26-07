import numpy as np
import trimesh

def load_data():
    pcd = trimesh.load('data/pointcloud10.obj', file_type = 'obj', force='pointcloud')
    points = np.array(pcd.vertices)
    
    return points
