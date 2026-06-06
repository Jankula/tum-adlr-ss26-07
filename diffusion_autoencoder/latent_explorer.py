import torch
import torch.multiprocessing as mp
from encoder import Encoder
from decoder import Decoder
import dataset
import utils
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


# def interactive_latent_explorer(z_0, decoder, num_ddim_steps=50):
def interactive_latent_explorer(z_0, input_queue, output_queue, num_ddim_steps=50):
    """Creates an interactive Open3D application to explore a diffusion VAE latent space."""
    
    # Force baseline reference and active tracking to flat CPU numpy arrays
    if isinstance(z_0, torch.Tensor):
        z_0_np = z_0.detach().cpu().numpy().astype(np.float64).flatten()
    else:
        z_0_np = np.array(z_0, dtype=np.float64).flatten()

    z_current = z_0_np.copy()
    latent_dim = z_current.shape[0]

    # Cleanly bypass WebRTC notebook background servers if present
    try:
        import open3d.visualization.webrtc_server as webrtc
        webrtc.disable_webrtc()
    except Exception:
        pass

    # --- Core Application Instantiation ---
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window(
        "Latent Space Explorer", 1200, 800
    )

    em = window.theme.font_size
    margin = int(0.5 * em)

    # --- UI Elements Initialization ---
    dir_input = gui.TextEdit()
    dir_input.placeholder_text = f"e.g., 0,1,-0.5... (Len: {latent_dim})"

    step_slider = gui.Slider(gui.Slider.DOUBLE)
    step_slider.set_limits(0.001, 2.0)
    step_slider.double_value = 0.1

    status_label = gui.Label("Current Position: z_0 base")

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

    def get_direction_vector():
        text = dir_input.text_value.strip()
        if not text:
            v = np.zeros(latent_dim)
            v[0] = 1.0
            return v
        try:
            v = np.fromstring(text, sep=",")
            if v.shape[0] != latent_dim:
                padded = np.zeros(latent_dim)
                padded[: min(v.shape[0], latent_dim)] = v[:latent_dim]
                return padded
            return v
        except Exception:
            v = np.zeros(latent_dim)
            v[0] = 1.0
            return v


    # --- Core Render Update Function ---
    def update_mesh_view():
        status_label.text = "GPU is decoding shape... please wait..."
        window.post_redraw()

        # 1. Send the current latent vector and steps to the background GPU process
        input_queue.put((z_current, num_ddim_steps))

        # 2. Wait for the GPU to return the NumPy array (This safely pauses the UI for a second)
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

        norm_from_origin = np.linalg.norm(z_current - z_0_np)
        status_label.text = f"Distance from z_0: {norm_from_origin:.4f}"

    # --- Button Event Listeners ---
    def on_step_forward():
        nonlocal z_current
        direction = get_direction_vector()
        norm = np.linalg.norm(direction)
        if norm > 1e-7:
            direction = direction / norm
        z_current += direction * step_slider.double_value
        update_mesh_view()

    def on_step_backward():
        nonlocal z_current
        direction = get_direction_vector()
        norm = np.linalg.norm(direction)
        if norm > 1e-7:
            direction = direction / norm
        z_current -= direction * step_slider.double_value
        update_mesh_view()

    def on_reset():
        nonlocal z_current
        z_current = z_0_np.copy()
        update_mesh_view()

    # --- Layout Container Assembly ---
    panel = gui.Vert(margin, gui.Margins(margin, margin, margin, margin))
    panel.add_child(gui.Label("1. Movement Target Coordinate (CSV)"))
    panel.add_child(dir_input)
    panel.add_child(gui.Label("2. Latent Step Size Slider"))
    panel.add_child(step_slider)
    panel.add_child(gui.Label("3. Execute Dynamic Decode"))
    
    button_layout = gui.Horiz(margin)
    btn_back = gui.Button(" <  Move Negative")
    btn_forward = gui.Button("Move Positive  > ")
    btn_back.set_on_clicked(on_step_backward)
    btn_forward.set_on_clicked(on_step_forward)
    button_layout.add_child(btn_back)
    button_layout.add_child(btn_forward)
    panel.add_child(button_layout)

    panel.add_fixed(int(em))
    panel.add_child(status_label)

    btn_reset = gui.Button("Reset to z_0")
    btn_reset.set_on_clicked(on_reset)
    panel.add_child(btn_reset)

    # CRITICAL FIX FOR UBUNTU DISPLAY DEVICE ENGINE:
    # Explicitly attach UI nodes directly onto the viewport window instance
    window.add_child(scene_widget)
    window.add_child(panel)

    # Separate bounds calculation inside the OS-driven context layer tracker
    def on_layout(layout_context):
        r = window.content_rect
        panel_width = 320
        panel.frame = gui.Rect(r.x, r.y, panel_width, r.height)
        scene_widget.frame = gui.Rect(
            r.x + panel_width, r.y, r.width - panel_width, r.height
        )

    window.set_on_layout(on_layout)

    # Seed the viewport context map
    update_mesh_view()

    # Calculate bounding parameters *after* geometry pipeline is fully seated
    bounds = scene_widget.scene.bounding_box
    scene_widget.setup_camera(60, bounds, [0, 0, 0])

    # Enforce global UI focus pipeline refresh
    window.post_redraw()

    # Hand off blocking execution safely to the Linux operating system manager thread
    gui.Application.instance.run()


def gpu_diffusion_worker(input_queue, output_queue):
    """Background process dedicated entirely to the Nvidia GPU."""
    # 1. Initialize CUDA inside the isolated process
    device = torch.device('cuda:0')
    
    encoder = Encoder(number_points=2048, in_channels=3, hidden_channels=64, latent_dim=32, clamp=False)
    decoder = Decoder(number_points=2048, point_dim=3, hidden_dim=128, latent_dim=32, timesteps=1000, beta_start=1e-4, beta_end=0.02)
    
    experiment_name = "Jun04_02-43_3_Objects_50epochs_16batch_KL0.001_hidden_dec128_time_emb"
    utils.reload_model_old(encoder, decoder, None, None, experiment_name, 'best', device)
    
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    # Signal the main thread that the GPU is loaded and ready
    output_queue.put("READY")

    # 2. Listen for incoming latent codes from the UI
    while True:
        request = input_queue.get()
        if request is None:
            break  # Poison pill to safely shutdown the process
        
        z_current, num_ddim_steps = request
        z_tensor = torch.from_numpy(z_current).float().view(1, -1).to(device)
        
        with torch.no_grad():
            from decoder import sample_ddim
            points_tensor = sample_ddim(decoder, z_tensor, 2048, num_ddim_steps)
            points = points_tensor.squeeze(0).cpu().numpy()
        
        # Send the generated point cloud back to the UI
        output_queue.put(points)


def main():
    # Force 'spawn' method to ensure clean CUDA isolation on Linux
    mp.set_start_method('spawn', force=True)
    
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    
    print("Spawning isolated PyTorch GPU Worker...")
    worker = mp.Process(target=gpu_diffusion_worker, args=(input_queue, output_queue))
    worker.start()
    
    # Wait for the model to finish loading into VRAM
    output_queue.get() 
    print("GPU Worker Ready! Extracting baseline latent code...")
    
    # Extract the z_0 baseline on the CPU so we don't dirty the main thread's GPU context
    # device_cpu = torch.device('cpu')
    # trainset = dataset.Dataset('overfit', timesteps=1000)
    # encoder_cpu = Encoder(number_points=2048, in_channels=3, hidden_channels=64, latent_dim=32, clamp=False)
    # utils.reload_model(encoder_cpu, None, None, None, "Jun04_02-43_3_Objects_50epochs_16batch_KL0.001_hidden_dec128_time_emb", 'best', device_cpu)
    # encoder_cpu.eval()

    # Extract the z_0 baseline on the CPU so we don't dirty the main thread's GPU context
    device_cpu = torch.device('cpu')
    trainset = dataset.Dataset('overfit', timesteps=1000)
    encoder_cpu = Encoder(number_points=2048, in_channels=3, hidden_channels=64, latent_dim=32, clamp=False)
    
    # Initialize a dummy CPU decoder just to satisfy utils.reload_model
    decoder_cpu = Decoder(number_points=2048, point_dim=3, hidden_dim=128, latent_dim=32, timesteps=1000, beta_start=1e-4, beta_end=0.02)
    
    # Pass the dummy decoder instead of None
    utils.reload_model_old(encoder_cpu, decoder_cpu, None, None, "Jun04_02-43_3_Objects_50epochs_16batch_KL0.001_hidden_dec128_time_emb", 'best', device_cpu)
    
    encoder_cpu.eval()
    
    # with torch.no_grad():
    #     mean, _ = encoder_cpu(trainset[0].unsqueeze(0))

    with torch.no_grad():
        mean, _ = encoder_cpu(trainset[0].unsqueeze(0).to(device_cpu))
    
    print("Launching Open3D UI...")
    # Launch UI, passing the queues instead of the decoder model!
    interactive_latent_explorer(mean, input_queue, output_queue, num_ddim_steps=50)
    
    # Cleanup when the UI window is closed
    input_queue.put(None)
    worker.join()
    print("Processes shutdown cleanly.")

if __name__ == "__main__":
    main()