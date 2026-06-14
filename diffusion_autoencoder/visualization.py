import torch
import open3d as o3d
import numpy as np


def visualize_diffusion_progress(samples_list, window_name="Diffusion Process"):
    """
    Takes a list of point clouds (from t=T to t=0) and animates them.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=800, height=600)
    
    # Create the initial geometry object
    pcd = o3d.geometry.PointCloud()
    # Convert first sample (noise) to Open3D format
    initial_points = samples_list[0].squeeze(0).cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(initial_points)
    
    # Optional: Give it a color (TUM Blue: 0, 0.4, 0.7)
    pcd.paint_uniform_color([0.0, 0.4, 0.7])
    
    vis.add_geometry(pcd)
    
    for i, sample in enumerate(samples_list):
        points = sample.squeeze(0).cpu().numpy()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Update the renderer
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        
    vis.run()
    vis.destroy_window()

# To use this, modify your sample loop to save intermediate steps:
# intermediate_samples = []
# for i in reversed(range(1000)):
#     ...
#     if i % 10 == 0: # Save every 10th step to keep memory low
#         intermediate_samples.append(x.detach().clone())


def visualize_comparison(pc_target, pc_generated, window_name="Target (Red) vs Generated (Blue)"):
    """
    pc_target: Tensor or Array of shape [N, 3]
    pc_generated: Tensor or Array of shape [N, 3]
    """
    # 1. Convert to NumPy and remove batch dimension if present
    if isinstance(pc_target, torch.Tensor):
        pc_target = pc_target.detach().cpu().squeeze().numpy()
    if isinstance(pc_generated, torch.Tensor):
        pc_generated = pc_generated.detach().cpu().squeeze().numpy()

    # 2. Create Open3D PointCloud objects
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(pc_target)
    # Paint it Red [R, G, B]
    target_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    gen_pcd = o3d.geometry.PointCloud()
    gen_pcd.points = o3d.utility.Vector3dVector(pc_generated)
    # Paint it Blue [R, G, B]
    gen_pcd.paint_uniform_color([0.0, 0.0, 1.0])

    # 3. Create a coordinate frame for orientation (optional but helpful)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # 4. Visualize
    o3d.visualization.draw_geometries([target_pcd, gen_pcd, coord_frame], 
                                      window_name=window_name)

# Example usage:
# visualize_comparison(target_shape, final_generated_shape)


def visualize_mesh_vs_points(mesh_target, pc_generated, window_name="Mesh vs. Points"):
    """
    mesh_target: An open3d.geometry.TriangleMesh object or path to .obj/.ply
    pc_generated: Tensor or Array of shape [N, 3]
    """
    # 1. Handle Mesh Input
    if isinstance(mesh_target, str):
        mesh = o3d.io.read_triangle_mesh(mesh_target)
    else:
        mesh = mesh_target
    
    mesh.compute_vertex_normals() # Ensure proper lighting
    # Paint it Red and make it slightly transparent
    mesh.paint_uniform_color([1.0, 0.7, 0.7]) 

    # 2. Handle Generated Points
    if isinstance(pc_generated, torch.Tensor):
        pc_generated = pc_generated.detach().cpu().squeeze().numpy()
    
    gen_pcd = o3d.geometry.PointCloud()
    gen_pcd.points = o3d.utility.Vector3dVector(pc_generated)
    gen_pcd.paint_uniform_color([0.0, 0.0, 1.0]) # Blue

    # 3. Visualization Setup
    # We use a visualizer object to customize the "Material" (transparency)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    
    vis.add_geometry(mesh)
    vis.add_geometry(gen_pcd)
    
    # Optional: Set rendering options for better visibility
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.point_size = 3.0
    
    print("Visualizing: Target Mesh is LIGHT RED, Generated Points are BLUE.")
    vis.run()
    vis.destroy_window()


def visualize_mesh_comparison(mesh_1, mesh_2, window_name="Mesh Comparison: Red vs Blue"):
    """
    mesh_1: Target mesh (Open3D object or path)
    mesh_2: Generated/Comparison mesh (Open3D object or path)
    """
    # 1. Load meshes if paths are provided
    def load_mesh(m):
        if isinstance(m, str):
            return o3d.io.read_triangle_mesh(m)
        return m

    m1 = load_mesh(mesh_1)
    m2 = load_mesh(mesh_2)

    # 2. Prepare Mesh 1 (Red)
    m1.compute_vertex_normals()
    m1.paint_uniform_color([1.0, 0.1, 0.1]) # Red

    # 3. Prepare Mesh 2 (Blue)
    m2.compute_vertex_normals()
    m2.paint_uniform_color([0.1, 0.1, 1.0]) # Blue

    # 4. Setup Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1024, height=768)
    
    vis.add_geometry(m1)
    vis.add_geometry(m2)

    # 5. Rendering Options for better comparison
    opt = vis.get_render_option()
    opt.mesh_show_wireframe = True  # Wireframe helps see overlapping surfaces
    opt.background_color = np.asarray([0.05, 0.05, 0.05]) # Dark background
    opt.light_intensity = 0.8
    
    print("Controls:")
    print("  'W' - Toggle Wireframe")
    print("  'L' - Toggle Lighting")
    print("  'R' - Reset View")
    
    vis.run()
    vis.destroy_window()

# Example usage:
# visualize_mesh_comparison("target_car.obj", "generated_car.obj")