import torch
import torch.multiprocessing as mp
import argparse
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from encoder import Encoder
from decoder import Decoder
import dataset
import utils

def interactive_latent_explorer(z1, z2, input_queue, output_queue, num_ddim_steps=50):
    """Creates an interactive Open3D application to interpolate between two latent vectors."""
    
    # Force baseline reference and active tracking to flat CPU numpy arrays
    def format_latent(z):
        if isinstance(z, torch.Tensor):
            return z.detach().cpu().numpy().astype(np.float64).flatten()
        return np.array(z, dtype=np.float64).flatten()

    z1_np = format_latent(z1)
    z2_np = format_latent(z2)
    
    current_step = 0

    # Cleanly bypass WebRTC notebook background servers if present
    try:
        import open3d.visualization.webrtc_server as webrtc
        webrtc.disable_webrtc()
    except Exception:
        pass

    # --- Core Application Instantiation ---
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window(
        "Latent Space Interpolation", 1200, 800
    )

    em = window.theme.font_size
    margin = int(0.5 * em)

    # --- UI Elements Initialization ---
    steps_input = gui.TextEdit()
    steps_input.text_value = "4"
    steps_input.placeholder_text = "Enter total steps (e.g., 10)"

    status_label = gui.Label("Current Position: z1")

    # --- 3D Viewport Setup ---
    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])  # White Theme

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=[0, 0, 0]
    )
    scene_widget.scene.add_geometry(
        "coord_frame", coord_frame, rendering.MaterialRecord()
    )

    def get_total_steps():
        """Safely extract the total number of steps from the text field."""
        try:
            val = int(steps_input.text_value.strip())
            return max(1, val) # Ensure at least 1 step
        except ValueError:
            return 10

    # --- Core Render Update Function ---
    def update_mesh_view():
        nonlocal current_step
        num_steps = get_total_steps()
        
        # Clamp current_step if user dynamically lowered the total step count
        if current_step > num_steps:
            current_step = num_steps
            
        t = current_step / num_steps
        z_current = z1_np + t * (z2_np - z1_np)

        status_label.text = f"Status: Interpolating... (t={t:.2f})"
        window.post_redraw()

        # 1. Send the current latent vector and steps to the background GPU process
        input_queue.put((z_current, num_ddim_steps))

        # 2. Wait for the GPU to return the NumPy array
        points = output_queue.get()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Height gradient metric mapping (Z-axis visualization colorization)
        colors = np.zeros_like(points)
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        if z_max - z_min > 1e-5:
            colors[:, 2] = (points[:, 2] - z_min) / (z_max - z_min)  # Blue depth
            colors[:, 0] = 1.0 - colors[:, 2]  # Red shift
        else:
            colors[:] = [0.0, 0.8, 1.0]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if scene_widget.scene.has_geometry("point_cloud"):
            scene_widget.scene.remove_geometry("point_cloud")

        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 5.0
        scene_widget.scene.add_geometry("point_cloud", pcd, material)

        status_label.text = f"Position: Step {current_step}/{num_steps} | Distance to z2: {np.linalg.norm(z_current - z2_np):.4f}"

    # --- Button Event Listeners ---
    def on_step_forward():
        nonlocal current_step
        num_steps = get_total_steps()
        if current_step < num_steps:
            current_step += 1
            update_mesh_view()
        else:
            print("Already at z2. Right button locked.")

    def on_step_backward():
        nonlocal current_step
        if current_step > 0:
            current_step -= 1
            update_mesh_view()
        else:
            print("Already at z1. Left button locked.")

    # --- Layout Container Assembly ---
    panel = gui.Vert(margin, gui.Margins(margin, margin, margin, margin))
    
    panel.add_child(gui.Label("1. Total Interpolation Steps (z1 -> z2)"))
    panel.add_child(steps_input)
    panel.add_child(gui.Label("2. Step Navigation"))
    
    button_layout = gui.Horiz(margin)
    btn_back = gui.Button(" <  Step to z1")
    btn_forward = gui.Button("Step to z2  > ")
    
    btn_back.set_on_clicked(on_step_backward)
    btn_forward.set_on_clicked(on_step_forward)
    
    button_layout.add_child(btn_back)
    button_layout.add_child(btn_forward)
    panel.add_child(button_layout)

    panel.add_fixed(int(em))
    panel.add_child(status_label)

    # Explicitly attach UI nodes directly onto the viewport window instance
    window.add_child(scene_widget)
    window.add_child(panel)

    def on_layout(layout_context):
        r = window.content_rect
        panel_width = 340
        panel.frame = gui.Rect(r.x, r.y, panel_width, r.height)
        scene_widget.frame = gui.Rect(
            r.x + panel_width, r.y, r.width - panel_width, r.height
        )

    window.set_on_layout(on_layout)

    # Seed the viewport context map with z1 initial state
    update_mesh_view()

    bounds = scene_widget.scene.bounding_box
    scene_widget.setup_camera(60, bounds, [0, 0, 0])
    window.post_redraw()

    # Hand off blocking execution safely
    gui.Application.instance.run()


def gpu_diffusion_worker(input_queue, output_queue, experiment_name, model_type, timesteps):
    """Background process dedicated entirely to the Nvidia GPU."""
    device = torch.device('cuda:0')
    
    encoder = Encoder(number_points=2048, in_channels=3, hidden_channels=64, latent_dim=32, clamp=False)
    decoder = Decoder(number_points=2048, point_dim=3, hidden_dim=128, latent_dim=32, timesteps=1000, beta_start=1e-4, beta_end=0.02)
    
    # Passing None values safely for objects that utils.reload_model will overwrite or ignore
    _, _, encoder, decoder = utils.reload_model(None, None, experiment_name, model_type, device)
    
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    output_queue.put("READY")

    while True:
        request = input_queue.get()
        if request is None:
            break  # Poison pill to safely shutdown
        
        z_current, num_ddim_steps = request
        z_tensor = torch.from_numpy(z_current).float().view(1, -1).to(device)
        
        with torch.no_grad():
            from decoder import sample_ddim
            points_tensor = sample_ddim(decoder, z_tensor, 2048, num_ddim_steps, timesteps)
            points = points_tensor.squeeze(0).cpu().numpy()
        
        output_queue.put(points)


def main(experiment_name, model_type, split_type, object_index1, object_index2):
    
    # Force 'spawn' method to ensure clean CUDA isolation on Linux
    mp.set_start_method('spawn', force=True)
    
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    
    
    # Extract the z1 and z2 baselines on the CPU
    device_cpu = torch.device('cpu')
    trainset = dataset.Dataset(split_type, timesteps=1000)
    
    encoder_cpu = Encoder(number_points=2048, in_channels=3, hidden_channels=64, latent_dim=32, clamp=False)
    decoder_cpu = Decoder(number_points=2048, point_dim=3, hidden_dim=128, latent_dim=32, timesteps=1000, beta_start=1e-4, beta_end=0.02)
    
    config, _, encoder_cpu, _ = utils.reload_model(None, None, experiment_name, model_type, device_cpu)
    
    timesteps = config['timesteps']

    print("Spawning isolated PyTorch GPU Worker...")
    worker = mp.Process(target=gpu_diffusion_worker, args=(input_queue, output_queue, experiment_name, model_type, timesteps))
    worker.start()
    
    # Wait for the model to finish loading into VRAM
    output_queue.get() 
    print("GPU Worker Ready! Extracting baseline latent codes...")
    
    encoder_cpu.eval()
    
    with torch.no_grad():
        print(f"Encoding object 1 (Index: {object_index1})...")
        z1, _ = encoder_cpu(trainset[object_index1].unsqueeze(0).to(device_cpu))
        
        print(f"Encoding object 2 (Index: {object_index2})...")
        z2, _ = encoder_cpu(trainset[object_index2].unsqueeze(0).to(device_cpu))
    
    print("Launching Open3D UI...")
    interactive_latent_explorer(z1, z2, input_queue, output_queue, num_ddim_steps=50)
    
    # Cleanup when the UI window is closed
    input_queue.put(None)
    worker.join()
    print("Processes shutdown cleanly.")


if __name__ == "__main__":
    # Standardize input injection via argparse for reusability from the CLI
    parser = argparse.ArgumentParser(description="Latent Space Interpolation Viewer")
    parser.add_argument("--experiment_name", type=str, default="Jun06_19-58_8_Objects_100timestep_1000epochs_16batch_KL0.001_hidden_dec128_time_emb")
    parser.add_argument("--type", type=str, default="best")
    parser.add_argument("--split_type", type=str, default="train")
    parser.add_argument("--object_index1", type=int, default=0, help="Dataset index for starting latent vector z1")
    parser.add_argument("--object_index2", type=int, default=1, help="Dataset index for target latent vector z2")
    
    args = parser.parse_args()
    
    main(args.experiment_name, args.type, args.split_type, args.object_index1, args.object_index2)