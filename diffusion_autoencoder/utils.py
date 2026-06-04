import os
import torch
import open3d as o3d
import numpy as np
import pathlib

def create_folders(hparams : dict):
    path = 'models'
    if not os.path.exists(os.path.join(path, f'model_{hparams["hidden1_size"]}_{hparams["hidden2_size"]}')) :
      os.mkdir(os.path.join(path, f'model_{hparams["hidden1_size"]}_{hparams["hidden2_size"]}'))
    if not os.path.exists(os.path.join(path, f'model_{hparams["hidden1_size"]}_{hparams["hidden2_size"]}', f'run{hparams["run"]}')) :
     os.mkdir(os.path.join(path, f'model_{hparams["hidden1_size"]}_{hparams["hidden2_size"]}', f'run{hparams["run"]}'))
    
    checkpoint_save_path = os.path.join(path, f'model_{hparams["hidden1_size"]}_{hparams["hidden2_size"]}', f'run{hparams["run"]}')

    path = 'logs'
    if not os.path.exists(os.path.join(path, f'model_{hparams["hidden1_size"]}_{hparams["hidden2_size"]}')) :
       os.mkdir(os.path.join(path, f'model_{hparams["hidden1_size"]}_{hparams["hidden2_size"]}'))
    if not os.path.exists(os.path.join(path, f'model_{hparams["hidden1_size"]}_{hparams["hidden2_size"]}', f'run{hparams["run"]}')) :
        os.mkdir(os.path.join(path, f'model_{hparams["hidden1_size"]}_{hparams["hidden2_size"]}', f'run{hparams["run"]}'))
    if not os.path.exists(os.path.join(path, f'model_{hparams["hidden1_size"]}_{hparams["hidden2_size"]}', f'run{hparams["run"]}', 'train')) : 
       os.mkdir(os.path.join(path, f'model_{hparams["hidden1_size"]}_{hparams["hidden2_size"]}', f'run{hparams["run"]}', 'train'))
    if not os.path.exists(os.path.join(path, f'model_{hparams["hidden1_size"]}_{hparams["hidden2_size"]}', f'run{hparams["run"]}', 'val')) : 
       os.mkdir(os.path.join(path, f'model_{hparams["hidden1_size"]}_{hparams["hidden2_size"]}', f'run{hparams["run"]}', 'val'))

    log_save_path_train = os.path.join(path, f'model_{hparams["hidden1_size"]}_{hparams["hidden2_size"]}', f'run{hparams["run"]}', 'train')
    log_save_path_val = os.path.join(path, f'model_{hparams["hidden1_size"]}_{hparams["hidden2_size"]}', f'run{hparams["run"]}', 'val')


    return checkpoint_save_path, log_save_path_train, log_save_path_val


def load_checkpoint(checkpoint_save_path, model, optimizer, scheduler, device):
    
    if os.path.exists(os.path.join(checkpoint_save_path, "checkpoint.pth")) :
        checkpoint = torch.load(os.path.join(checkpoint_save_path, "checkpoint.pth"), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
        model.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        return checkpoint.get('epoch', 0)

    return 0

def best_model_validation_loss(checkpoint_save_path, device):
    if os.path.exists(os.path.join(checkpoint_save_path, "best_model_checkpoint.pth")) :
        best_model_checkpoint = torch.load(os.path.join(checkpoint_save_path, "best_model_checkpoint.pth"), map_location=device)
        return best_model_checkpoint.get('loss', 100)
    
    return 100


def save_model(encoder, decoder, optimizer, scheduler, config, model_config, type):
    checkpoint = {
    'encoder_state': encoder.state_dict(),
    'decoder_state': decoder.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'scheduler_state': scheduler.state_dict(),
    'config': config,
    'model_config': model_config
    }
    path = pathlib.Path(f'models/{config["experiment_name"]}')
    path.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, pathlib.Path(f'models/{config["experiment_name"]}/{type}.pt'))

def reload_model(encoder, decoder, optimizer, scheduler, experiment_name, type, device):
    checkpoint = torch.load(pathlib.Path(f'models/{experiment_name}/{type}.pt'), weights_only=True, map_location=device)

    encoder.load_state_dict(checkpoint['encoder_state'])
    decoder.load_state_dict(checkpoint['decoder_state'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    config = checkpoint['config']
    
    model_config = checkpoint['model_config']

    encoder.eval()
    decoder.eval()

    return config, model_config


def count_parameters(model):
    # Total parameters (including non-trainable ones)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Only parameters that require gradients
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def model_memory_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


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