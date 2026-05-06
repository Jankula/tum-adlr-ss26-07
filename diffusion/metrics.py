from scipy.spatial import cdist
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import KDTree
import numpy as np

def chamfer_distance(p1, p2):
    # Calculate pairwise distances
    dist_matrix = cdist(p1, p2, metric='euclidean')
    
    # Distance from p1 to closest p2
    d1 = np.mean(np.min(dist_matrix, axis=1))
    # Distance from p2 to closest p1
    d2 = np.mean(np.min(dist_matrix, axis=0))
    
    return d1 + d2

def hausdorff_distance(p1, p2):
    # directed_hausdorff returns (distance, index1, index2)
    d1 = directed_hausdorff(p1, p2)[0]
    d2 = directed_hausdorff(p2, p1)[0]
    return max(d1, d2)

def fast_chamfer(p1, p2):
    tree1 = KDTree(p1)
    tree2 = KDTree(p2)
    
    dist1, _ = tree2.query(p1) # Closest in p2 for each in p1
    dist2, _ = tree1.query(p2) # Closest in p1 for each in p2
    
    return np.mean(dist1**2) + np.mean(dist2**2)