import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt

def visualize_aggregate_clouds(data_dir, max_objects, grasps_per_object):
    data_path = Path(data_dir)
    npz_files = list(data_path.rglob("*.npz"))
    
    if not npz_files:
        print(f"No .npz files found in {data_dir}")
        return

    print(f"Found {len(npz_files)} files. Loading...")

    # Limit the number of objects to prevent running out of RAM
    if max_objects:
        npz_files = npz_files[:max_objects]

    geometries = []
    
    # Use a colormap to give each object a distinct color
    cmap = plt.get_cmap("tab20")

    for i, file_path in enumerate(npz_files):
        data = np.load(file_path)
        
        if "point_clouds" not in data:
            continue
            
        # Shape: [num_grasps, num_points, 3]
        pcs = data["point_clouds"]
        
        # Limit how many grasps we show per object to reduce visual clutter
        pcs = pcs[:grasps_per_object]
        
        # Flatten the grasps down to a single array of points for Open3D: [total_points, 3]
        flat_pcs = pcs.reshape(-1, 3)
        
        # Create the Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(flat_pcs)
        
        # Assign a unique color to this specific object
        color = cmap(i % 20)[:3]  # Grab RGB from the colormap
        pcd.paint_uniform_color(color)
        
        geometries.append(pcd)

    print(f"Successfully loaded {len(geometries)} objects.")

    # Add a coordinate frame at the origin [0,0,0] to verify centering
    # The axes are: X (Red), Y (Green), Z (Blue)
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5, origin=[0, 0, 0])
    geometries.append(origin_frame)

    print("Opening Open3D viewer. Use your mouse to rotate and scroll wheel to zoom.")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Aggregate Point Cloud Dataset",
        width=1280,
        height=720,
        left=50,
        top=50,
        point_show_normal=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize all point clouds in Open3D.")
    
    parser.add_argument("--data-dir", type=str, default="/home/nikola/Projects/tum-adlr-ss26-07/diffusion_autoencoder/data/train", help="Path to the split directory containing .npz files.")
    parser.add_argument("--max-objects", type=int, default=50, help="Maximum number of objects to load (prevents crashing).")
    parser.add_argument("--grasps-per-object", type=int, default=1, help="How many grasps to render per object. 1 is usually enough for shape visualization.")
    
    args = parser.parse_args()
    
    visualize_aggregate_clouds(args.data_dir, args.max_objects, args.grasps_per_object)