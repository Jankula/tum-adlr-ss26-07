import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

def verify_dataset_normalization(data_dir, norm_type, sample_limit=None, plot=False):
    """
    Iterates through the dataset and mathematically verifies if the point clouds 
    adhere to the expected normalization bounds.
    """
    data_path = Path(data_dir)
    npz_files = list(data_path.rglob("*.npz")) # Searches recursively in train/val/test
    
    if not npz_files:
        print(f"No .npz files found in {data_dir}")
        return

    print(f"Found {len(npz_files)} files. Verifying '{norm_type}' normalization...")
    
    # Trackers for global statistics
    all_means = []
    all_max_norms = []
    all_variances = []
    coord_variances = []
    
    # Optional: collect a subset of points for plotting so we don't blow up the RAM
    plot_points = []

    for i, file_path in enumerate(tqdm(npz_files)):
        if sample_limit and i >= sample_limit:
            break
            
        # Load point cloud (shape: [num_grasps, num_points, 3])
        data = np.load(file_path)
        if "point_clouds" not in data:
            continue
            
        pcs = data["point_clouds"]
        
        # Calculate empirical stats for this specific file
        file_mean = np.mean(pcs, axis=(0, 1))
        file_max_norm = np.max(np.linalg.norm(pcs, axis=2))
        file_var = np.var(pcs)
        file_coord_var = np.var(pcs, axis=(0, 1))
        
        all_means.append(file_mean)
        all_max_norms.append(file_max_norm)
        all_variances.append(file_var)
        coord_variances.append(file_coord_var)
        
        if plot:
            # Flatten and sample 10% of points from this file to keep memory safe
            flat_pcs = pcs.reshape(-1, 3)
            sampled_idx = np.random.choice(flat_pcs.shape[0], size=int(flat_pcs.shape[0]*0.1), replace=False)
            plot_points.append(flat_pcs[sampled_idx])

    # Aggregate global empirical statistics
    global_mean = np.mean(all_means, axis=0)
    absolute_max_norm = np.max(all_max_norms)
    avg_variance = np.mean(all_variances)
    avg_coord_var = np.mean(coord_variances, axis=0)

    # Print Verification Results
    print("\n" + "="*50)
    print("NORMALIZATION VERIFICATION RESULTS")
    print("="*50)
    print(f"Empirical Global Mean: {global_mean} (Should be very close to [0, 0, 0])")
    
    if norm_type == "max_norm":
        print(f"Empirical Max Norm: {absolute_max_norm:.6f} (Should be exactly 1.0)")
        if np.isclose(absolute_max_norm, 1.0, atol=1e-3):
            print("✅ PASSED: Data is strictly bounded within the unit sphere.")
        else:
            print("❌ FAILED: Data exceeds max norm bounds!")
            
    elif norm_type == "global_var":
        print(f"Empirical Global Variance: {avg_variance:.6f} (Should be close to 1.0)")
        if np.isclose(avg_variance, 1.0, atol=1e-1):
            print("✅ PASSED: Data variance is properly scaled.")
        else:
            print("❌ WARNING: Variance deviates from expected unit variance.")
            
    elif norm_type == "coord_var":
        print(f"Empirical Coordinate Variance [X, Y, Z]: {avg_coord_var}")
        print("(All three values should be close to 1.0)")
        if np.allclose(avg_coord_var, [1.0, 1.0, 1.0], atol=1e-1):
            print("✅ PASSED: Independent coordinate variance is properly scaled.")
        else:
            print("❌ WARNING: Coordinate variances deviate from 1.0.")

    # Plotting
    if plot and plot_points:
        print("\nGenerating coordinate histograms...")
        all_points = np.vstack(plot_points)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axis_labels = ['X', 'Y', 'Z']
        colors = ['red', 'green', 'blue']
        
        for i in range(3):
            axes[i].hist(all_points[:, i], bins=100, color=colors[i], alpha=0.7)
            axes[i].set_title(f'{axis_labels[i]} Coordinate Distribution')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            
            # Add vertical line at 0 to verify centering
            axes[i].axvline(0, color='black', linestyle='dashed', linewidth=2)
            
            # Add bounds for max_norm
            if norm_type == "max_norm":
                axes[i].axvline(-1, color='gray', linestyle='dotted', linewidth=2)
                axes[i].axvline(1, color='gray', linestyle='dotted', linewidth=2)

        plt.suptitle(f"Point Cloud Distributions ({norm_type})")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check mathematical bounds of normalized point clouds.")
    parser.add_argument("--data_dir", type=str, default='/home/nikola/Projects/tum-adlr-ss26-07/diffusion_autoencoder/data/train', help="Path to the dataset directory (e.g., the output_dir from preprocessing).")
    parser.add_argument("--norm-type", type=str, choices=["max_norm", "global_var", "coord_var"], default="max_norm")
    parser.add_argument("--plot", action="store_true", help="Generate and show histograms of the coordinates.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Stop checking after N files (for quick tests).")
    
    args = parser.parse_args()
    
    verify_dataset_normalization(args.data_dir, args.norm_type, args.sample_limit, args.plot)