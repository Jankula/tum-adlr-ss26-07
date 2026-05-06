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