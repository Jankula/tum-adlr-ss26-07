# libraries

import open3d as o3d
import trimesh
import numpy as np
from pathlib import Path
import os
import random

def get_project_root():
    """Finds the root by looking for a marker file."""
    current = Path.cwd()
    # Look upwards for the root marker
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "requirements.txt").exists():
            return parent
    return current # Fallback to CWD

def get_random_shapenet_meshes(seed: int, num_objects: int, root: Path) -> list[Path]:
    """
    Navigates the ShapeNet hierarchy to return a random list of mesh paths.
    
    Hierarchy: root/data/studentGrasping/student_grasps_v1/shapenet_id/sub_id/0-9/mesh.obj
    """
    # 1. Define the base search directory
    # Adjust 'root' to your actual absolute or relative path
    base_path = root / Path("data/studentGrasping/student_grasps_v1")
    
    if not base_path.exists():
        raise FileNotFoundError(f"The path {base_path} does not exist.")

    # 2. Find all mesh.obj files within the hierarchy
    # We use rglob to recursively find all mesh.obj files 
    # located inside the 0-9 folders
    all_meshes = list(base_path.rglob("**/[0-9]/mesh.obj"))
    
    if not all_meshes:
        print("No mesh.obj files found in the specified hierarchy.")
        return []

    # 3. Set the random seed for reproducibility
    random.seed(seed)
    
    # 4. Handle cases where requested number exceeds available meshes
    k = min(num_objects, len(all_meshes))
    
    # 5. Sample and return
    return random.sample(all_meshes, k)

def normalize_points_max(pcs):
    # Centering: Subtract the mean of the points
    centroid = np.mean(pcs, axis=0)
    pcs = pcs - centroid
    
    # Scaling: Find the furthest point and divide by that distance
    m = np.max(np.sqrt(np.sum(pcs**2, axis=1)))
    pcs = pcs / m
    return pcs

#it scales the x and y and z each, changes the overall shape
def normalize_statistically_points(pcs):
    # Centering: Subtract the mean of the points
    centroid = np.mean(pcs, axis=0)
    pcs = pcs - centroid
    
    # Scaling: Find the furthest point and divide by that distance
    v = np.var(pcs, axis=0)
    return pcs / np.sqrt(v)

def get_exterior_points(points, radius_multiplier=10.0):
    """
    points: (N, 3) numpy array
    radius_multiplier: Controls the 'spherical flip' projection. 
                       Usually 10x the max extent of the object.
    """
    # 1. Convert to Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 2. Define Camera Positions 
    # We use the 8 corners of a cube + 6 axis points to cover all angles
    camera_locations = [
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0], [0, 0, 2], [0, 0, -2]
    ]
    
    # Scale cameras so they are well outside the normalized unit sphere
    camera_locations = np.array(camera_locations) * 5.0 
    
    # This set will store indices of points that are visible from ANY camera
    visible_indices = set()
    
    # 3. The Loop: View from each location
    # radius for HPR should be larger than the point cloud extents
    hpr_radius = np.max(np.linalg.norm(points, axis=1)) * radius_multiplier

    for cam_pos in camera_locations:
        # _, pt_map returns the indices of points visible from cam_pos
        _, pt_map = pcd.hidden_point_removal(cam_pos, hpr_radius)
        visible_indices.update(pt_map)
    
    # 4. Filter the points
    exterior_indices = list(visible_indices)
    exterior_pcd = pcd.select_by_index(exterior_indices)
    
    return np.asarray(exterior_pcd.points)

def points_to_mesh_bpa(points, radius=None):
    """
    Reconstructs a mesh using Ball Pivoting Algorithm (BPA).
    Note: Requires normals.
    """
    # 1. Convert to Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 2. Estimate Normals (Crucial for reconstruction)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(10)
    
    # 3. Define the ball radii 
    # If not provided, we calculate them based on average distance
    if radius is None:
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = [avg_dist, avg_dist * 2] # Use multiple radii for better coverage
    
    # 4. Run Ball Pivoting
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radius)
    )
    
    return mesh