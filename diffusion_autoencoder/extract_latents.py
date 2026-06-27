import torch
from pathlib import Path
import numpy as np
import pathlib
from tqdm import tqdm
from encoder import Encoder
from decoder import Decoder
from utils import reload_model



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    RAW_DATA_DIR = Path("/home/nikola/tum-adlr-ss26-07/diffusion_autoencoder/data")
    LEAN_DATA_DIR = Path("/home/nikola/tum-adlr-ss26-07/diffusion_autoencoder/lean_data")
    
    experiment_name = "Jun23_11-47_7000epochs_128latent_enc128_dec512_globalnorm"
    print(f"Loading pre-trained checkpoint: {experiment_name}...")
    config, model_config, encoder, _ = reload_model(None, None, experiment_name, 'best_train', device)

    splits = ["train", "val", "test"]
    
    for split in splits:
        src_split_dir = RAW_DATA_DIR / split
        dst_split_dir = LEAN_DATA_DIR / split
        
        if not src_split_dir.exists():
            print(f"Skipping split '{split}' (directory not found: {src_split_dir})")
            continue
            
        # Recreate mirror output folder cleanly
        dst_split_dir.mkdir(parents=True, exist_ok=True)
        
        file_paths = sorted(list(src_split_dir.glob("*.npz")), key=lambda x: x.name)        
        print(f"\nProcessing {len(file_paths)} files in '{split}' split...")

        for fp in tqdm(file_paths, desc=f"Encoding {split}"):
            # 1. Read data from original compressed archive
            with np.load(fp) as data:
                point_clouds = data["point_clouds"]  # Shape: [num_grasps, 2048, 3]
                joint_angles = data["joint_angles"]  # Shape: [num_grasps, 12]
            
            num_grasps = point_clouds.shape[0]
            computed_codes = []

            # 2. Extract codes batch-wise per object grasp configuration
            with torch.no_grad():
                for grasp_idx in range(num_grasps):
                    # Prepare input tensor: [1, 2048, 3]
                    pc_tensor = torch.tensor(point_clouds[grasp_idx], dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # Compute distributions and sample code vector
                    mean, log_variance = encoder(pc_tensor)
                    
                    # Store back as a clean numpy array on CPU: shape [latent_dim]
                    computed_codes.append(mean.squeeze(0).cpu().numpy())

            # Stack individual grasp codes into matrix shape: [num_grasps, latent_dim]
            np_codes = np.stack(computed_codes, axis=0)

            # 3. Save to identical file name in lean_data folder
            output_filepath = dst_split_dir / fp.name
            np.savez_compressed(
                output_filepath,
                codes=np_codes,                  # Matches Dataset_Latent_grasp_and_code expected key
                joint_angles=joint_angles        # Retained unchanged
            )

    print("\nProcessing Complete! Lean datasets are generated successfully.")

if __name__ == "__main__":
    main()