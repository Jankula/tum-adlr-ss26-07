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

    if max_objects:
        npz_files = npz_files[:max_objects]

    geometries = []
    cmap = plt.get_cmap("tab20")

    for i, file_path in enumerate(npz_files):
        data = np.load(file_path)
        if "point_clouds" not in data:
            continue
            
        pcs = data["point_clouds"][:grasps_per_object]
        flat_pcs = pcs.reshape(-1, 3)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(flat_pcs)
        
        color = cmap(i % 20)[:3]
        pcd.paint_uniform_color(color)
        
        geometries.append(pcd)

    print(f"Successfully loaded {len(geometries)} objects.")

    # ==========================================
    # VISUAL BOUNDARIES FOR NORMALIZATION VERIFICATION
    # ==========================================

    # 1. Coordinate Axes (Length exactly 1.0)
    # X = Red, Y = Green, Z = Blue. The tips represent exactly 1.0
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    geometries.append(origin_frame)

    # 2. Wireframe Unit Bounding Box (spanning from -1.0 to 1.0)
    bbox_points = [
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [-1, 1,  1], [1, 1,  1]
    ]
    bbox_lines = [
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    bbox = o3d.geometry.LineSet()
    bbox.points = o3d.utility.Vector3dVector(bbox_points)
    bbox.lines = o3d.utility.Vector2iVector(bbox_lines)
    bbox.paint_uniform_color([0.3, 0.3, 0.3]) # Dark Grey
    geometries.append(bbox)

    # 3. Wireframe Unit Sphere (Radius exactly 1.0)
    # Converted to a LineSet so it renders as a transparent cage, not a solid ball
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    sphere_lines = o3d.geometry.LineSet.create_from_triangle_mesh(sphere_mesh)
    sphere_lines.paint_uniform_color([0.7, 0.7, 0.7]) # Light Grey
    geometries.append(sphere_lines)

    print("Opening Open3D viewer. Use your mouse to rotate and scroll wheel to zoom.")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Aggregate Point Cloud Dataset with Boundaries",
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
    parser.add_argument("--grasps-per-object", type=int, default=5, help="How many grasps to render per object. 1 is usually enough for shape visualization.")
    
    args = parser.parse_args()
    
    visualize_aggregate_clouds(args.data_dir, args.max_objects, args.grasps_per_object)