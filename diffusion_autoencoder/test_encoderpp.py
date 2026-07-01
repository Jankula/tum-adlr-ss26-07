import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import PointCloudDataset, get_point_cloud_files
from encoder import LocalPointNetEncoder
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

test_path = "../data/preprocessed/test"
test_point_clouds_ds = PointCloudDataset(get_point_cloud_files(test_path), number_points=512)
test_point_clouds = DataLoader(test_point_clouds_ds, batch_size=10, drop_last=True)

encoder = LocalPointNetEncoder(number_points=512, in_channels=3, num_patches=8, points_per_patch=64,
                              local_latent_dim=16, hidden_channels=64, latent_dim=128)

test_point_clouds = iter(test_point_clouds)
pcs = next(test_point_clouds)
print("shape pcs: ", pcs.shape)

centers = encoder.farthest_point_sampling(pcs, 8)
print(centers.shape)
centers = encoder.index_points(pcs, centers)
print("shape centers: ", centers.shape)

pc = pcs[0].cpu().numpy()
print("shape pc: ", pc.shape)
center = centers[0].cpu().numpy()
print("center shape: ", center.shape)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], color="b", label="original point cloud")
ax.scatter(center[:, 0], center[:, 1], center[:, 2], color="r", s=100, label="centers")
ax.grid("off")
plt.show()