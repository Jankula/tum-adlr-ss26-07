import torch

def chamfer_distance(p1, p2):
    """
    Computes Chamfer Distance between two point clouds p1 and p2.
    p1: (B, N, 3) - batch, num_points_1, coordinates
    p2: (B, M, 3) - batch, num_points_2, coordinates
    """
    # 1. Expand dimensions to compute pairwise distances
    # p1: (B, N, 1, 3), p2: (B, 1, M, 3)
    p1 = p1.unsqueeze(2)
    p2 = p2.unsqueeze(1)
    
    # 2. Compute squared distances: (B, N, M)
    dist = torch.sum((p1 - p2) ** 2, dim=-1)
    
    # 3. For each point in p1, find the distance to the nearest neighbor in p2
    dl_min, _ = torch.min(dist, dim=2)  # (B, N)
    
    # 4. For each point in p2, find the distance to the nearest neighbor in p1
    dr_min, _ = torch.min(dist, dim=1)  # (B, M)
    
    # 5. Average the distances
    chamfer_dist = torch.mean(dl_min, dim=1) + torch.mean(dr_min, dim=1)
    
    return chamfer_dist