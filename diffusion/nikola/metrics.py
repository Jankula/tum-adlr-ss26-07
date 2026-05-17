from pytorch3d.loss import chamfer_distance
import numpy as np
import torch

# def chamfer_distance_custom(p1, p2):
#     # Calculate pairwise distances
#     dist_matrix = cdist(p1, p2, metric='euclidean')
    
#     # Distance from p1 to closest p2
#     d1 = np.mean(np.min(dist_matrix, axis=1))
#     # Distance from p2 to closest p1
#     d2 = np.mean(np.min(dist_matrix, axis=0))
    
#     return d1 + d2

# def hausdorff_distance(p1, p2):
#     # directed_hausdorff returns (distance, index1, index2)
#     d1 = directed_hausdorff(p1, p2)[0]
#     d2 = directed_hausdorff(p2, p1)[0]
#     return max(d1, d2)

# def fast_chamfer(p1, p2):
#     tree1 = KDTree(p1)
#     tree2 = KDTree(p2)
    
#     dist1, _ = tree2.query(p1) # Closest in p2 for each in p1
#     dist2, _ = tree1.query(p2) # Closest in p1 for each in p2
    
#     return np.mean(dist1**2) + np.mean(dist2**2)

def SNR(mse_loss_per_item, alpha_cumprod, timestep, gamma = 5.0):
    """
        Computes the Min-SNR Gamma reweighted loss for a batch.
        
        Args:
            mse_loss_per_item (torch.Tensor): Un-reduced MSE loss per batch element 
                                            with shape [B].
            alpha_cumprod (torch.Tensor): The full pre-calculated cumulative alpha buffer 
                                        from the diffuser, shape [B].
            timestep (torch.Tensor): The sampled batch of timesteps, shape [B].
            gamma (float): The SNR clamping threshold hyperparameter. Default is 5.0.
            
        Returns:
            torch.Tensor: A single scalar tensor representing the weighted batch loss.
    """
    alpha_bar = alpha_cumprod[timestep]
    snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
    mse_loss_weights = torch.clamp(snr, max=gamma) / snr

    SNR_loss = (mse_loss_per_item * mse_loss_weights).mean()
    
    return SNR_loss


# def Chamfer_loss():
#     # Chamfer Loss

#     alpha_cumprod = diffuser.alpha_cumprod[t].view(-1, 1, 1)
#     pred_x0 = (noisy_pc - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
    
#     return chamfer(pred_x0, target_pc)[0]

def predict_x0_and_chamfer(noisy_pc, predicted_noise, target_pc, alpha_cumprod, t):
    """
    Algebraically reconstructs x_0 from the current diffusion state 
    and calculates the raw geometric Chamfer Distance.
    
    Args:
        noisy_pc (torch.Tensor): Noisy point cloud batch [B, N, 3]
        predicted_noise (torch.Tensor): Model's noise prediction [B, N, 3]
        target_pc (torch.Tensor): Ground truth clean point cloud batch [B, N, 3]
        alpha_cumprod (torch.Tensor): The registered alpha_cumprod buffer [Timesteps]
        t (torch.Tensor): Sampled batch timesteps [B]
        
    Returns:
        tuple: (raw_chamfer_loss, pred_x0)
    """
    # 1. Align alpha dimension for batch broadcasting: [B] -> [B, 1, 1]
    alpha_bar = alpha_cumprod[t].view(-1, 1, 1)
    
    # 2. Add epsilon clamping to prevent division-by-zero anomalies at t=999
    sqrt_alpha = torch.sqrt(alpha_bar).clamp(min=1e-5)
    sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_bar)
    
    # 3. Solve the forward diffusion equation for x_0: 
    # x_t = sqrt(alpha)*x_0 + sqrt(1-alpha)*noise
    pred_x0 = (noisy_pc - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha
    
    # 4. Compute the mathematical coordinate-to-surface distance
    per_item_chamfer, _ = chamfer_distance(pred_x0, target_pc, batch_reduction = None)
    
    return per_item_chamfer, pred_x0

def compute_damped_geometry_loss(noisy_pc, predicted_noise, target_pc, alpha_cumprod, timestep, max_timesteps=1000):
    """
    Highly optimized wrapper that computes the raw chamfer distance and 
    applies linear timestep damping entirely on the GPU.
    
    Args:
        noisy_pc (torch.Tensor): Noisy point cloud batch [B, N, 3]
        predicted_noise (torch.Tensor): Model's noise prediction [B, N, 3]
        target_pc (torch.Tensor): Clean ground-truth shapes [B, N, 3]
        alpha_cumprod (torch.Tensor): Diffuser schedule buffer [Timesteps]
        t (torch.Tensor): Sampled batch timesteps [B]
        max_timesteps (int): Total diffusion steps (usually 1000)
        
    Returns:
        tuple: (damped_chamfer_loss_scalar, raw_chamfer_loss_scalar)
    """
    # 1. Get raw geometry stats from the clean function
    per_item_chamfer, pred_x0 = predict_x0_and_chamfer(noisy_pc, predicted_noise, target_pc, alpha_cumprod, timestep)
    
    # 2. GPU-side Vectorized Damping Calculation
    # No .item() or .cpu() calls here. We keep it as a pure GPU tensor.
    damping_weight = 1.0 - (timestep.float() / max_timesteps)
    
    # 3. Apply the damping scalar to the raw batch average
    damped_chamfer = per_item_chamfer * damping_weight
    
    return damped_chamfer.mean(), per_item_chamfer.mean(), pred_x0