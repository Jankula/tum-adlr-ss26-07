import argparse
import json
from pathlib import Path
import pybullet as p
import numpy as np

def visualize_processed_data(npz_path, urdf_path, stats_path, norm_type):
    """
    Visualizes the generated point clouds directly in PyBullet alongside the hand.
    Reverses the global normalization so the object aligns with the hand frame.
    """
    print(f"Loading processed data from: {npz_path}")
    data = np.load(npz_path)
    point_clouds = data["point_clouds"] # Shape: [num_grasps, num_points, 3]
    joint_angles = data["joint_angles"]
    scores = data["scores"]

    num_grasps = point_clouds.shape[0]
    num_points = point_clouds.shape[1]

    # 1. Reverse the Normalization (if applied)
    if norm_type != "none" and stats_path:
        print(f"Reversing '{norm_type}' normalization using {stats_path}...")
        with open(stats_path, "r") as f:
            stats = json.load(f)
            
        mean = np.array(stats["global_mean"])
        
        if norm_type == "max_norm":
            scale = stats["max_norm"]
        elif norm_type == "global_var":
            scale = np.sqrt(stats["global_variance"])
        elif norm_type == "coord_var":
            scale = np.sqrt(np.array(stats["coord_variance"]))
            
        # Reverse math: original = (normalized * scale) + mean
        point_clouds = (point_clouds * scale) + mean
    else:
        print("Skipping normalization reversal (data assumed unnormalized).")

    # 2. Setup PyBullet and Load Fixed Hand
    p.connect(p.GUI)
    hand_id = p.loadURDF(
        str(urdf_path),
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1], # Fixed at origin
        useFixedBase=True,
        flags=p.URDF_MAINTAIN_LINK_ORDER,
    )

    # Make the point colors red for high visibility against the hand
    point_colors = [[0.8, 0.2, 0.2] for _ in range(num_points)]

    # Track the unique ID of the point cloud to explicitly delete it
    current_point_cloud_id = None

    # 3. Iterate through grasps
    for i in range(num_grasps):
        
        # Explicitly remove the previous point cloud if it exists
        if current_point_cloud_id is not None:
            p.removeUserDebugItem(current_point_cloud_id)

        grasp_joints = joint_angles[i]
        score = scores[i]

        # Set finger joint angles
        for k, j in enumerate([1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21]):
            p.resetJointState(hand_id, jointIndex=j, targetValue=grasp_joints[k], targetVelocity=0)
            # Set coupled joint
            if j in [3, 9, 15, 21]:
                p.resetJointState(hand_id, jointIndex=j + 1, targetValue=grasp_joints[k], targetVelocity=0)

        # Draw the point cloud AND save its unique ID
        pts = point_clouds[i].tolist()
        current_point_cloud_id = p.addUserDebugPoints(
            pointPositions=pts,
            pointColorsRGB=point_colors,
            pointSize=4.0 
        )

        print(f"[{i+1}/{num_grasps}] Visualizing Grasp | Score: {score:.4f}")
        input("Press Enter in terminal to view the next grasp...")

    print("Finished visualizing all grasps in this file.")
    p.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Processed Point Cloud Grasps")
    
    # Needs to point to a specific generated .npz file
    parser.add_argument("npz_path", type=str, help="Path to a processed .npz file in the train/val/test splits.")
    
    parser.add_argument("--urdf-path", type=str, default="/home/nikola/Projects/tum-adlr-ss26-07/data/studentGrasping/urdfs/dlr2.urdf", help="Path to the hand URDF.")
    parser.add_argument("--stats-path", type=str, default="/home/nikola/Projects/tum-adlr-ss26-07/diffusion_autoencoder/data/normalization_stats.json", help="Path to the generated JSON stats file.")
    parser.add_argument("--norm-type", type=str, choices=["none", "max_norm", "global_var", "coord_var"], default="max_norm", help="The normalization type that was used during generation.")
    
    args = parser.parse_args()

    visualize_processed_data(
        npz_path=Path(args.npz_path),
        urdf_path=Path(args.urdf_path),
        stats_path=Path(args.stats_path) if args.stats_path else None,
        norm_type=args.norm_type
    )