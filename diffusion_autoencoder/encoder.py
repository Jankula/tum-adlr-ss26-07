import torch.nn as nn
import torch.nn.functional as F
import torch

 
class Resnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size),
        )

        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()

    def forward(self, x):
        return F.relu(self.resblock(x) + self.skip(x))
    

class PointNetEncoder(nn.Module):
    def __init__(self, number_points=2048, in_channels=3, hidden_channels=128, latent_dim=256, clamp=False):
        super().__init__()
        self.number_points = number_points
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.clamp = clamp

        #Shared MLP idea from Pointnet with Conv1d, does not weight order of points in pointcloud
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1), #Conv1d expects [B, C, N]
            nn.BatchNorm1d(hidden_channels), 
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 2 * hidden_channels, 1),
            nn.BatchNorm1d(2 * hidden_channels),
            nn.Conv1d(2 * hidden_channels, 4 * hidden_channels, 1)
        )

        self.max_pool = nn.MaxPool1d(kernel_size=number_points)

        #For calculating the mean of the latent variable
        self.fc1_mean = nn.Linear(4 * hidden_channels, 2 * hidden_channels)
        self.fc2_mean = nn.Linear(2 * hidden_channels, hidden_channels)
        self.fc3_mean = nn.Linear(hidden_channels, latent_dim)

        self.fc1_mean_norm = nn.BatchNorm1d(2 * hidden_channels)
        self.fc2_mean_norm = nn.BatchNorm1d(hidden_channels)
        self.fc3_mean_norm = nn.BatchNorm1d(latent_dim)

        #For calculating the log-variance of the latent variable
        self.fc1_var = nn.Linear(4 * hidden_channels, 2 * hidden_channels)
        self.fc2_var = nn.Linear(2 * hidden_channels, hidden_channels)
        self.fc3_var = nn.Linear(hidden_channels, latent_dim)

        self.fc1_var_norm = nn.BatchNorm1d(2 * hidden_channels)
        self.fc2_var_norm = nn.BatchNorm1d(hidden_channels)
        self.fc3_var_norm = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        x = x.transpose(1, 2) # Transform from [B, N, C] to [B, C, N]
        B, C, N = x.shape

        x = self.projection(x)
        x = self.max_pool(x)
        x = x.view(-1, 4 * self.hidden_channels)

        #Reparametrization trick, to enable backpropagation training of the VAE
        #Model learns mean and log_variance of the Gaussian latent space distribution
        #Sampling of latents handled via outside function with learned mean and log_variance
        mean = F.relu(self.fc1_mean_norm(self.fc1_mean(x)))
        mean = F.relu(self.fc2_mean_norm(self.fc2_mean(mean)))
        mean = self.fc3_mean_norm(self.fc3_mean(mean))

        log_variance = F.relu(self.fc1_var_norm(self.fc1_var(x)))
        log_variance = F.relu(self.fc2_var_norm(self.fc2_var(log_variance)))
        log_variance = self.fc3_var_norm(self.fc3_var(log_variance))

        if self.clamp:
            log_variance = torch.clamp(log_variance, min=-30.0, max=20.0) # For numerical stability

        return mean, log_variance
    

class PointNetEncoder_old(nn.Module):
    def __init__(self, number_points=2048, in_channels=3, hidden_channels=128, latent_dim=256, clamp=False):
        super().__init__()
        self.number_points = number_points
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.clamp = clamp

        #Shared MLP idea from Pointnet with Conv1d, does not weight order of points in pointcloud
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1), #Conv1d expects [B, C, N]
            nn.GroupNorm(8, hidden_channels), #GroupNorm for small batch_size training, Batchnorm weak to noisy batch statistics
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 2 * hidden_channels, 1),
            nn.GroupNorm(8, 2 * hidden_channels),
            nn.ReLU(),
            nn.Conv1d(2 * hidden_channels, 4 * hidden_channels, 1),
        )

        self.max_pool = nn.MaxPool1d(kernel_size=number_points)

        #For calculating the mean of the latent variable
        self.fc1_mean = nn.Linear(4 * hidden_channels, 2 * hidden_channels)
        self.fc2_mean = nn.Linear(2 * hidden_channels, hidden_channels)
        self.fc3_mean = nn.Linear(hidden_channels, latent_dim)

        self.fc1_mean_norm = nn.LayerNorm(2 * hidden_channels)
        self.fc2_mean_norm = nn.LayerNorm(hidden_channels)
        self.fc3_mean_norm = nn.LayerNorm(latent_dim)

        #For calculating the log-variance of the latent variable
        self.fc1_var = nn.Linear(4 * hidden_channels, 2 * hidden_channels)
        self.fc2_var = nn.Linear(2 * hidden_channels, hidden_channels)
        self.fc3_var = nn.Linear(hidden_channels, latent_dim)

        self.fc1_var_norm = nn.LayerNorm(2 * hidden_channels)
        self.fc2_var_norm = nn.LayerNorm(hidden_channels)
        self.fc3_var_norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        x = x.transpose(1, 2) # Transform from [B, N, C] to [B, C, N]
        B, C, N = x.shape

        x = self.projection(x)
        x = self.max_pool(x)
        x = x.view(-1, 4 * self.hidden_channels)

        #Reparametrization trick, to enable backpropagation training of the VAE
        #Model learns mean and log_variance of the Gaussian latent space distribution
        #Sampling of latents handled via outside function with learned mean and log_variance
        mean = F.silu(self.fc1_mean_norm(self.fc1_mean(x)))
        mean = F.silu(self.fc2_mean_norm(self.fc2_mean(mean)))
        mean = self.fc3_mean(mean)

        log_variance = F.silu(self.fc1_var_norm(self.fc1_var(x)))
        log_variance = F.silu(self.fc2_var_norm(self.fc2_var(log_variance)))
        log_variance = self.fc3_var(log_variance)

        if self.clamp:
            log_variance = torch.clamp(log_variance, min=-30.0, max=20.0) # For numerical stability

        return mean, log_variance
    

class ResnetDecoder(nn.Module):
    def __init__(self, number_points, latent_dim):
        super().__init__()
        self.number_points = number_points
        self.latent_dim = latent_dim

        self.upsample1 = nn.Linear(latent_dim, 2 * latent_dim)
        self.norm1 = nn.LayerNorm(2 * latent_dim)
        self.resblock1 = Resnet(2 * latent_dim, 2 * latent_dim, 2 * latent_dim)
        self.upsample2 = nn.Linear(2 * latent_dim, 4 * latent_dim)
        self.norm2 = nn.LayerNorm(4 * latent_dim)
        self.resblock2 = Resnet(4 * latent_dim, 4 * latent_dim, 4 * latent_dim)
        self.output = nn.Linear(4 * latent_dim, number_points * 3)

    def forward(self, x):
        B, _ = x.shape
        x = F.relu(self.norm1(self.upsample1(x)))
        x = self.resblock1(x)
        x = F.relu(self.norm2(self.upsample2(x)))
        x = self.resblock2(x)
        return self.output(x).view(B, self.number_points, 3) #[B, N, C]
    
class PointnetVAE(nn.Module):
    def __init__(self, number_points=2048, in_channels=3, hidden_channels=64, latent_dim=256):
        super().__init__()
        self.encoder = PointNetEncoder(number_points, in_channels, hidden_channels, latent_dim, clamp=False)
        self.decoder = ResnetDecoder(number_points, latent_dim)

    def sample_latent_z(self, mean, log_variance):
        std = torch.exp(0.5 * log_variance) # compute std from log_variance
        eps = torch.randn_like(mean)
        return mean + eps * std # encoder latent z
    
    def forward(self, x):
        mean, log_variance = self.encoder(x)
        z = self.sample_latent_z(mean, log_variance)
        return self.decoder(z), mean, log_variance # Reconstructed Point Cloud + mean and log_variance for KLD loss
    


class LocalPointNetEncoder(nn.Module):
    def __init__(
        self,
        number_points=512,
        in_channels=3,
        num_patches=8,
        points_per_patch=64,
        local_latent_dim=16,
        hidden_channels=64,
        latent_dim=128,
        use_center=True,
        clamp=False,
    ):
        super().__init__()

        self.number_points = number_points
        self.in_channels = in_channels
        self.num_patches = num_patches
        if num_patches < 4:
            self.num_patches = 4
        self.points_per_patch = points_per_patch
        self.local_latent_dim = local_latent_dim
        self.latent_dim = latent_dim
        self.use_center = use_center
        self.hidden_channels = hidden_channels
        self.clamp = clamp

        # relative xyz + optional center xyz
        local_input_dim = in_channels + (3 if use_center else 0)

        self.local_pointnet = nn.Sequential(
            nn.Conv1d(local_input_dim, hidden_channels, 1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),

            nn.Conv1d(hidden_channels, 2 * hidden_channels, 1),
            nn.BatchNorm1d(2 * hidden_channels),
            nn.ReLU(),

            nn.Conv1d(2 * hidden_channels, 4 * hidden_channels, 1),
            nn.BatchNorm1d(4 * hidden_channels),
            nn.ReLU(),

            nn.Conv1d(4 * hidden_channels, 2 * hidden_channels, 1),
            nn.BatchNorm1d(2 * hidden_channels),
            nn.ReLU(),
            nn.Conv1d(2 * hidden_channels, hidden_channels, 1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, local_latent_dim, 1),

        )

        # Falls num_patches * local_latent_dim != latent_dim,
        # mappe sauber auf gewünschten latent_dim.
        self.global_projection_mean = nn.Sequential(
            nn.Linear(self.num_patches * local_latent_dim * (self.points_per_patch + 1),  4 * latent_dim),
            nn.LayerNorm(4 * latent_dim),
            nn.ReLU(),
            nn.Linear(4 * latent_dim, 2 * latent_dim),
            nn.LayerNorm(2 * latent_dim),
            nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim)
        )
        
        self.global_projection_var = nn.Sequential(
            nn.Linear(self.num_patches * local_latent_dim * (self.points_per_patch + 1), 4 * latent_dim),
            nn.LayerNorm(4 * latent_dim),
            nn.ReLU(),
            nn.Linear(4 * latent_dim, 2 * latent_dim),
            nn.LayerNorm(2 * latent_dim),
            nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim)
        )
        
    
    def farthest_point_sampling(self, x, num_centers):
        """
        x: [B, N, 3]
        returns indices: [B, num_centers]
        """
        B, N, _ = x.shape
        device = x.device

        centers = torch.zeros(B, num_centers, dtype=torch.long, device=device)
        distances = torch.full((B, N), float("inf"), device=device)

        farthest = torch.randint(0, N, (B,), device=device)
        batch_indices = torch.arange(B, device=device)

        for i in range(num_centers):
            centers[:, i] = farthest
            centroid = x[batch_indices, farthest].view(B, 1, 3)
            dist = torch.sum((x - centroid) ** 2, dim=-1)
            distances = torch.minimum(distances, dist)
            farthest = torch.max(distances, dim=1)[1]

        return centers


    def index_points(self, x, idx):
        """
        x: [B, N, C]
        idx: [B, S] or [B, S, K]
        returns: [B, S, C] or [B, S, K, C]
        """
        B = x.shape[0]
        batch_indices = torch.arange(B, device=x.device).view(B, *([1] * (idx.dim() - 1)))
        batch_indices = batch_indices.expand_as(idx)
        return x[batch_indices, idx]


    def knn_group(self, x, centers, k):
        """
        x: [B, N, 3]
        centers: [B, S, 3]
        returns grouped points: [B, S, K, 3]
        """
        dists = torch.cdist(centers, x)  # [B, S, N]
        idx = dists.topk(k=k, dim=-1, largest=False)[1]  # [B, S, K]
        grouped = self.index_points(x, idx)  # [B, S, K, 3]
        return grouped

    def forward(self, x):
        """
        x: [B, N, 3]
        returns z: [B, latent_dim]
        S (num_patches)
        K (num_points_per_patch)
        D (point_dim)
        """
        B, N, C = x.shape

        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        assert N >= self.points_per_patch, "points_per_patch must be <= number of input points"

        # 1. FPS-Zentren wählen
        center_idx = self.farthest_point_sampling(x, self.num_patches)  # [B, S]
        centers = self.index_points(x, center_idx)  # [B, S, 3]

        # 2. KNN-Gruppen um Zentren bilden
        grouped = self.knn_group(x, centers, self.points_per_patch)  # [B, S, K, 3]

        # 3. Relative Koordinaten
        relative = grouped - centers.unsqueeze(2)  # [B, S, K, 3]

        if self.use_center:
            center_features = centers.unsqueeze(2).expand(-1, -1, self.points_per_patch, -1)
            local_input = torch.cat([relative, center_features], dim=-1)  # [B, S, K, 6]
        else:
            local_input = relative  # [B, S, K, 3]

        # 4. Shared Local PointNet auf alle Patches anwenden
        B, S, K, D = local_input.shape
        local_input = local_input.reshape(B * S, K, D)      # [B*S, K, D]
        local_input = local_input.transpose(1, 2)           # [B*S, D, K]

        local_features = self.local_pointnet(local_input)   # [B*S, local_latent_dim, K]
        max_features = torch.max(local_features, dim=-1)[0]  # [B*S, local_latent_dim]

        # 5. Lokale Latents konkatenieren
        max_features = max_features.unsqueeze(dim=-1) # [B*S, local_latent_dim, 1]
        local_features = torch.cat((local_features, max_features), dim=-1) # [B*S, local_latent_dim, K+1] 
        local_features = local_features.reshape(B, S * self.local_latent_dim * (K+1))  # [B, S * local_latent_dim * (K+1)]
        
        #local_features = local_features.transpose(1, 2)     # [B*S, K, local_latent_dim]
        #local_features = local_features.reshape(B, S * K * self.local_latent_dim)  # [B, S * K * local_latent_dim]

        # 6. Auf finalen Latent Space projizieren
        mean = self.global_projection_mean(local_features)  # [B, latent_dim]
        log_variance = self.global_projection_var(local_features)  # [B, latent_dim]
        
        if self.clamp:
            log_variance = torch.clamp(log_variance, min=-30.0, max=20.0) # For numerical stability

        return mean, log_variance

class LocalPointNetEncoder2(nn.Module):
    def __init__(
        self,
        number_points=512,
        in_channels=3,
        num_patches=8,
        points_per_patch=64,
        local_latent_dim=16,
        hidden_channels=64,
        latent_dim=128,
        use_center=True,
        clamp=False,
    ):
        super().__init__()

        self.number_points = number_points
        self.in_channels = in_channels
        self.num_patches = num_patches
        if num_patches < 4:
            self.num_patches = 4
        self.points_per_patch = points_per_patch
        self.local_latent_dim = local_latent_dim
        self.latent_dim = latent_dim
        self.use_center = use_center
        self.hidden_channels = hidden_channels
        self.clamp = clamp

        # relative xyz + optional center xyz
        self.local_input_dim = in_channels + (3 if use_center else 0)

        self.local_pointnet = nn.Sequential(
            nn.Linear(self.local_input_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),

            nn.Linear(hidden_channels, 2 * hidden_channels),
            nn.LayerNorm(2 * hidden_channels),
            nn.ReLU(),

            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),

            nn.Linear(hidden_channels, local_latent_dim),
        )

        # Falls num_patches * local_latent_dim != latent_dim,
        # mappe sauber auf gewünschten latent_dim.
        self.global_projection_mean = nn.Sequential(
            nn.Linear(self.num_patches * (local_latent_dim + 3) * (self.points_per_patch + 1),  4 * latent_dim),
            nn.LayerNorm(4 * latent_dim),
            nn.ReLU(),
            nn.Linear(4 * latent_dim, 2 * latent_dim),
            nn.LayerNorm(2 * latent_dim),
            nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim)
        )
        
        self.global_projection_var = nn.Sequential(
            nn.Linear(self.num_patches * (local_latent_dim + 3) * (self.points_per_patch + 1), 4 * latent_dim),
            nn.LayerNorm(4 * latent_dim),
            nn.ReLU(),
            nn.Linear(4 * latent_dim, 2 * latent_dim),
            nn.LayerNorm(2 * latent_dim),
            nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim)
        )
        
    
    def farthest_point_sampling(self, x, num_centers):
        """
        x: [B, N, 3]
        returns indices: [B, num_centers]
        """
        B, N, _ = x.shape
        device = x.device

        centers = torch.zeros(B, num_centers, dtype=torch.long, device=device)
        distances = torch.full((B, N), float("inf"), device=device)

        farthest = torch.randint(0, N, (B,), device=device)
        batch_indices = torch.arange(B, device=device)

        for i in range(num_centers):
            centers[:, i] = farthest
            centroid = x[batch_indices, farthest].view(B, 1, 3)
            dist = torch.sum((x - centroid) ** 2, dim=-1)
            distances = torch.minimum(distances, dist)
            farthest = torch.max(distances, dim=1)[1]

        return centers


    def index_points(self, x, idx):
        """
        x: [B, N, C]
        idx: [B, S] or [B, S, K]
        returns: [B, S, C] or [B, S, K, C]
        """
        B = x.shape[0]
        batch_indices = torch.arange(B, device=x.device).view(B, *([1] * (idx.dim() - 1)))
        batch_indices = batch_indices.expand_as(idx)
        return x[batch_indices, idx]


    def knn_group(self, x, centers, k):
        """
        x: [B, N, 3]
        centers: [B, S, 3]
        returns grouped points: [B, S, K, 3]
        """
        dists = torch.cdist(centers, x)  # [B, S, N]
        idx = dists.topk(k=k, dim=-1, largest=False)[1]  # [B, S, K]
        grouped = self.index_points(x, idx)  # [B, S, K, 3]
        return grouped

    def forward(self, x):
        """
        x: [B, N, 3]
        returns z: [B, latent_dim]
        S (num_patches)
        K (num_points_per_patch)
        D (point_dim)
        """
        B, N, C = x.shape

        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        assert N >= self.points_per_patch, "points_per_patch must be <= number of input points"

        # 1. FPS-Zentren wählen
        center_idx = self.farthest_point_sampling(x, self.num_patches)  # [B, S]
        centers = self.index_points(x, center_idx)  # [B, S, 3]

        # 2. KNN-Gruppen um Zentren bilden
        grouped = self.knn_group(x, centers, self.points_per_patch)  # [B, S, K, 3]

        # 3. Relative Koordinaten
        relative = grouped - centers.unsqueeze(2)  # [B, S, K, 3]

        if self.use_center:
            center_features = centers.unsqueeze(2).expand(-1, -1, self.points_per_patch, -1)
            local_input = torch.cat([relative, center_features], dim=-1)  # [B, S, K, 6]
        else:
            local_input = relative  # [B, S, K, 3]

        # 4. Shared Local PointNet auf alle Patches anwenden
        B, S, K, D = local_input.shape
        local_features = self.local_pointnet(local_input)   # [B, S, K, local_latent_dim]
        
        if self.use_center:
            local_features = torch.cat((local_features, center_features), dim=-1) # [B, S, K, local_latent_dim + 3]
            
        max_features = torch.max(local_features, dim=2)[0]  # [B, S, K, local_latent_dim + 3]

        # 5. Lokale Latents konkatenieren
        max_features = max_features.unsqueeze(dim=2) # [B, S, 1, local_latent_dim + 3]
        local_features = torch.cat((local_features, max_features), dim=2) # [B, S, K+1, local_latent_dim + 3] 
        local_features = local_features.reshape(B, S * (self.local_latent_dim+3) * (K+1))  # [B, S * (local_latent_dim + 3) * (K+1)]

        # 6. Auf finalen Latent Space projizieren
        mean = self.global_projection_mean(local_features)  # [B, latent_dim]
        log_variance = self.global_projection_var(local_features)  # [B, latent_dim]
        
        if self.clamp:
            log_variance = torch.clamp(log_variance, min=-30.0, max=20.0) # For numerical stability

        return mean, log_variance
    

class LocalPointNetEncoder3(nn.Module):
    def __init__(
        self,
        number_points=512,
        in_channels=3,
        num_patches=8,
        points_per_patch=64,
        local_latent_dim=16,
        hidden_channels=64,
        latent_dim=128,
        use_center=True,
        clamp=False,
    ):
        super().__init__()

        self.number_points = number_points
        self.in_channels = in_channels
        self.num_patches = num_patches
        if num_patches < 4:
            self.num_patches = 4
        self.points_per_patch = points_per_patch
        self.local_latent_dim = local_latent_dim
        self.latent_dim = latent_dim
        self.use_center = use_center
        self.hidden_channels = hidden_channels
        self.clamp = clamp

        # relative xyz + optional center xyz
        self.local_input_dim = in_channels + (3 if use_center else 0)

        self.local_pointnet = nn.Sequential(
            nn.Linear(self.local_input_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),

            nn.Linear(hidden_channels, 2 * hidden_channels),
            nn.LayerNorm(2 * hidden_channels),
            nn.ReLU(),

            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),

            nn.Linear(hidden_channels, local_latent_dim),
        )

        # Falls num_patches * local_latent_dim != latent_dim,
        # mappe sauber auf gewünschten latent_dim.
        self.global_projection_mean = nn.Sequential(
            nn.Linear(self.num_patches * (local_latent_dim + 3),  4 * latent_dim),
            nn.LayerNorm(4 * latent_dim),
            nn.ReLU(),
            nn.Linear(4 * latent_dim, 2 * latent_dim),
            nn.LayerNorm(2 * latent_dim),
            nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim)
        )
        
        self.global_projection_var = nn.Sequential(
            nn.Linear(self.num_patches * (local_latent_dim + 3), 4 * latent_dim),
            nn.LayerNorm(4 * latent_dim),
            nn.ReLU(),
            nn.Linear(4 * latent_dim, 2 * latent_dim),
            nn.LayerNorm(2 * latent_dim),
            nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim)
        )
        
    
    def farthest_point_sampling(self, x, num_centers):
        """
        x: [B, N, 3]
        returns indices: [B, num_centers]
        """
        B, N, _ = x.shape
        device = x.device

        centers = torch.zeros(B, num_centers, dtype=torch.long, device=device)
        distances = torch.full((B, N), float("inf"), device=device)

        farthest = torch.randint(0, N, (B,), device=device)
        batch_indices = torch.arange(B, device=device)

        for i in range(num_centers):
            centers[:, i] = farthest
            centroid = x[batch_indices, farthest].view(B, 1, 3)
            dist = torch.sum((x - centroid) ** 2, dim=-1)
            distances = torch.minimum(distances, dist)
            farthest = torch.max(distances, dim=1)[1]

        return centers


    def index_points(self, x, idx):
        """
        x: [B, N, C]
        idx: [B, S] or [B, S, K]
        returns: [B, S, C] or [B, S, K, C]
        """
        B = x.shape[0]
        batch_indices = torch.arange(B, device=x.device).view(B, *([1] * (idx.dim() - 1)))
        batch_indices = batch_indices.expand_as(idx)
        return x[batch_indices, idx]


    def knn_group(self, x, centers, k):
        """
        x: [B, N, 3]
        centers: [B, S, 3]
        returns grouped points: [B, S, K, 3]
        """
        dists = torch.cdist(centers, x)  # [B, S, N]
        idx = dists.topk(k=k, dim=-1, largest=False)[1]  # [B, S, K]
        grouped = self.index_points(x, idx)  # [B, S, K, 3]
        return grouped

    def forward(self, x):
        """
        x: [B, N, 3]
        returns z: [B, latent_dim]
        S (num_patches)
        K (num_points_per_patch)
        D (point_dim)
        """
        B, N, C = x.shape

        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        assert N >= self.points_per_patch, "points_per_patch must be <= number of input points"

        # 1. FPS-Zentren wählen
        center_idx = self.farthest_point_sampling(x, self.num_patches)  # [B, S]
        centers = self.index_points(x, center_idx)  # [B, S, 3]

        # 2. KNN-Gruppen um Zentren bilden
        grouped = self.knn_group(x, centers, self.points_per_patch)  # [B, S, K, 3]

        # 3. Relative Koordinaten
        relative = grouped - centers.unsqueeze(2)  # [B, S, K, 3]

        if self.use_center:
            center_features = centers.unsqueeze(2).expand(-1, -1, self.points_per_patch, -1)
            local_input = torch.cat([relative, center_features], dim=-1)  # [B, S, K, 6]
        else:
            local_input = relative  # [B, S, K, 3]

        # 4. Shared Local PointNet auf alle Patches anwenden
        B, S, K, D = local_input.shape
        local_features = self.local_pointnet(local_input)   # [B, S, K, local_latent_dim]
        
        if self.use_center:
            local_features = torch.cat((local_features, center_features), dim=-1) # [B, S, K, local_latent_dim + 3]
            
        local_features = torch.max(local_features, dim=2)[0]  # [B, S, local_latent_dim + 3]

        # 5. Lokale Latents konkatenieren
        #max_features = max_features.unsqueeze(dim=2) # [B, S, 1, local_latent_dim + 3]
        #local_features = torch.cat((local_features, max_features), dim=2) # [B, S, K+1, local_latent_dim + 3] 
        local_features = local_features.reshape(B, S * (self.local_latent_dim+3))  # [B, S * (local_latent_dim + 3)]

        # 6. Auf finalen Latent Space projizieren
        mean = self.global_projection_mean(local_features)  # [B, latent_dim]
        log_variance = self.global_projection_var(local_features)  # [B, latent_dim]
        
        if self.clamp:
            log_variance = torch.clamp(log_variance, min=-30.0, max=20.0) # For numerical stability

        return mean, log_variance
    
    