import argparse
import trimesh
import pathlib
import os
import shutil
import numpy as np
import subprocess
import json
import random
from multiprocessing import Pool
from tqdm import tqdm
from scipy.spatial.transform import Rotation

# ==========================================
# 1. Mesh Preprocessing & Transformation
# ==========================================

def make_watertight_and_simplify(mesh_path, manifold_bin, simplify_bin, tmp_dir, max_vertices):
    """Runs Manifold and Simplify on the base mesh."""
    pid = os.getpid()
    tmp_mani_path = tmp_dir / f"mani_{pid}.obj"
    tmp_simp_path = tmp_dir / f"simp_{pid}.obj"
    tmp_mani_path.unlink(missing_ok=True)
    tmp_simp_path.unlink(missing_ok=True)

    try:
        subprocess.run([manifold_bin, str(mesh_path), str(tmp_mani_path)], capture_output=True)
        if not tmp_mani_path.is_file(): return None
        
        subprocess.run([simplify_bin, "-i", str(tmp_mani_path), "-o", str(tmp_simp_path), "-m", "-f", str(max_vertices)], capture_output=True)
        if not tmp_simp_path.is_file(): return None

        mesh = trimesh.load(tmp_simp_path)
        return mesh
    except Exception as e:
        print(f"Error in manifold/simplify: {e}")
        return None
    finally:
        tmp_mani_path.unlink(missing_ok=True)
        tmp_simp_path.unlink(missing_ok=True)

def process_object(args):
    """Processes a single object: transforms to hand-centric frame and samples points for top grasps."""
    scale_dir, out_dir, manifold_bin, simplify_bin, max_vertices, center_by_com, num_grasps, num_points = args
    
    tmp_dir = pathlib.Path("/tmp/preprocess_student_grasp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Simplify base mesh
    mesh_path = scale_dir / "mesh.obj"
    mesh = make_watertight_and_simplify(mesh_path, manifold_bin, simplify_bin, tmp_dir, max_vertices)
    if mesh is None: return None

    # Center base mesh
    object_center_pos = mesh.center_mass if center_by_com else mesh.bounding_box.centroid
    mesh.apply_translation(-object_center_pos)

    # 2. Load and filter top grasps
    data = np.load(scale_dir / "recording.npz")
    sorted_idx = np.argsort(data["scores"])[::-1][:num_grasps] # Take top N grasps
    
    scores = data["scores"][sorted_idx]
    joint_angles = data["grasps"][sorted_idx, 7:]
    
    point_clouds = []
    
    # 3. Transform and sample per grasp
    for idx in sorted_idx:
        hand_pos = data["grasps"][idx, :3]
        hand_rot = Rotation.from_quat(data["grasps"][idx, 3:7], scalar_first=False)
        
        # Calculate object transformation relative to a fixed hand at origin
        inv_hand_rot = hand_rot.inv()
        object_pos = inv_hand_rot.apply(object_center_pos - hand_pos)
        
        # Copy mesh to apply specific transformation
        grasp_mesh = mesh.copy()
        
        # Apply rotation then translation to mesh
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = inv_hand_rot.as_matrix()
        transform_matrix[:3, 3] = object_pos
        grasp_mesh.apply_transform(transform_matrix)
        
        # Sample points directly from the transformed mesh
        points, _ = trimesh.sample.sample_surface(grasp_mesh, num_points)
        point_clouds.append(points)

    # Save to corresponding split directory
    obj_id = scale_dir.parent.name + "_" + scale_dir.name
    np.savez_compressed(
        out_dir / f"{obj_id}.npz",
        point_clouds=np.stack(point_clouds), # Shape: [num_grasps, num_points, 3]
        joint_angles=joint_angles,
        scores=scores
    )
    return True

# ==========================================
# 2. Global Normalization
# ==========================================

def compute_global_normalization(train_dir, val_dir):
    """Calculates normalization stats across the Train and Val sets."""
    print("Computing global normalization statistics...")
    all_points = []
    
    # Load all point clouds from Train and Val
    for split_dir in [train_dir, val_dir]:
        for file_path in split_dir.glob("*.npz"):
            data = np.load(file_path)
            all_points.append(data["point_clouds"]) # [grasps, points, 3]
            
    if not all_points:
        print("No data found to compute normalization.")
        return None
        
    global_pcs = np.concatenate(all_points, axis=0) # [total_grasps, points, 3]
    
    # 1. Mean
    global_mean = np.mean(global_pcs, axis=(0, 1))
    centered_pcs = global_pcs - global_mean
    
    # 2. Variances & Max Norm
    max_norm = float(np.max(np.linalg.norm(centered_pcs, axis=2)))
    global_variance = float(np.var(centered_pcs))
    coord_variance = np.var(centered_pcs, axis=(0, 1)).tolist()

    stats = {
        "global_mean": global_mean.tolist(),
        "max_norm": max_norm,
        "global_variance": global_variance,
        "coord_variance": coord_variance
    }
    
    # Save stats to root output dir
    stats_path = train_dir.parent / "normalization_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"Normalization stats saved to {stats_path}")
    return stats

def apply_global_normalization(output_dir, stats, norm_type):
    """Applies the chosen normalization method to all generated .npz files."""

    if norm_type == "none":
        print("Skipping normalization application (--normalize none selected).")
        return

    print(f"Applying '{norm_type}' normalization to all point clouds...")
    
    mean = np.array(stats["global_mean"])
    
    # We use standard deviation (sqrt of variance) to scale properly
    if norm_type == "max_norm":
        scale = stats["max_norm"]
    elif norm_type == "global_var":
        scale = np.sqrt(stats["global_variance"])
    elif norm_type == "coord_var":
        scale = np.sqrt(np.array(stats["coord_variance"]))
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")

    # Apply to Train, Val, and Test
    for split in ["train", "val", "test"]:
        split_dir = output_dir / split
        if not split_dir.exists(): continue
        
        for file_path in split_dir.glob("*.npz"):
            data = np.load(file_path)
            pcs = data["point_clouds"]
            joint_angles = data["joint_angles"]
            scores = data["scores"]
            
            # Center the points
            normalized_pcs = pcs - mean
            
            # Scale the points
            normalized_pcs = normalized_pcs / scale
            
            # Overwrite the file with normalized data
            np.savez_compressed(
                file_path,
                point_clouds=normalized_pcs,
                joint_angles=joint_angles,
                scores=scores
            )
            
    print("Normalization successfully applied to all files!")

# ==========================================
# 3. Main Pipeline Runner
# ==========================================

def build_dataset(args):
    base_dir = pathlib.Path(args.base_dir)
    output_dir = pathlib.Path(args.output_dir)
    manifold_bin = str(pathlib.Path(args.manifold_dir) / "manifold")
    simplify_bin = str(pathlib.Path(args.manifold_dir) / "simplify")

    # 1. Setup Directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    stats_file = output_dir / "normalization_stats.json"

    print("Clearing old data to prevent double-normalization...")
    # Delete the old stats file
    if stats_file.exists():
        stats_file.unlink() # Pathlib's way of deleting a single file

    for d in [train_dir, val_dir, test_dir]:
        if d.exists():
            shutil.rmtree(d) # Aggressively delete the folder and all old .npz files inside it
        d.mkdir(parents=True, exist_ok=True) # Recreate the fresh, empty folder

    # 2. Collect and Shuffle Object Directories
    object_dirs = []
    for category_dir in [p for p in base_dir.iterdir() if p.is_dir()]:
        for object_dir in [p for p in category_dir.iterdir() if p.is_dir()]:
            for scale_dir in [p for p in object_dir.iterdir() if p.is_dir()]:
                object_dirs.append(scale_dir)
                
    random.shuffle(object_dirs)
    n_total = len(object_dirs)
    
    # 3. Split Logic
    if args.process_all:
        n_train = int(n_total * args.train_ratio)
        n_val = int(n_total * args.val_ratio)
        n_test = n_total - n_train - n_val
    else:
        n_train = args.num_train or 0
        n_val = args.num_val or 0
        n_test = args.num_test or 0
        if n_train + n_val + n_test > n_total:
            n_train = min(n_train, n_total)
            n_val = min(n_val, n_total - n_train)
            n_test = min(n_test, n_total - n_train - n_val)

    splits = {
        "train": (object_dirs[:n_train], train_dir),
        "val": (object_dirs[n_train:n_train+n_val], val_dir),
        "test": (object_dirs[n_train+n_val:n_train+n_val+n_test], test_dir)
    }

    # 4. Build Processing Arguments
    args_list = []
    for split_name, (dirs, split_out_dir) in splits.items():
        for scale_dir in dirs:
            args_list.append((
                scale_dir, split_out_dir, manifold_bin, simplify_bin, 
                args.max_num_vertices, args.center_by_com, 
                args.num_grasps_per_object, args.num_sample_points
            ))

    if len(args_list) == 0:
        print("No objects selected for processing.")
        return

    # 5. Multiprocessing Execution
    num_workers = max(1, (os.cpu_count() or 1) - 1)
    print(f"Starting multiprocessing pool with {num_workers} workers...")
    success_count = 0
    with Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(process_object, args_list)
        for result in tqdm(results, total=len(args_list)):
            if result is not None:
                success_count += 1

    # 6. Compute Normalization & Apply it
    stats = compute_global_normalization(train_dir, val_dir)
    if stats is not None:
        apply_global_normalization(output_dir, stats, args.normalize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Hand-Centric Point Cloud Dataset.")
    
    parser.add_argument("--base-dir", type=str, default="/home/nikola/tum-adlr-ss26-07/data/studentGrasping/student_grasps_v1")
    parser.add_argument("--output-dir", type=str, default="/home/nikola/tum-adlr-ss26-07/diffusion_autoencoder/data")
    parser.add_argument("--manifold-dir", type=str, default="/home/nikola/Manifold/build")
    
    # Normalization Argument
    parser.add_argument("--normalize", type=str, choices=["none", "max_norm", "global_var", "coord_var"], default="max_norm",
                        help="Choose which normalization to apply to the final point clouds.")
    
    parser.add_argument("--process-all", action="store_true")

    parser.add_argument("--num-train", type=int, default=10)
    parser.add_argument("--num-val", type=int, default=2)
    parser.add_argument("--num-test", type=int, default=2)
    
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    
    parser.add_argument("--num-grasps-per-object", type=int, default=1)
    parser.add_argument("--num-sample-points", type=int, default=2048)
    parser.add_argument("--max-num-vertices", type=int, default=2048)
    
    parser.add_argument("--center-by-com", action="store_true")
    
    args = parser.parse_args()
    build_dataset(args)