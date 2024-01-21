import numpy as np
import open3d as o3d
import torch.nn as nn
from sklearn.neighbors import KDTree


class ChamferDistanceMetric(nn.Module):
    def forward(
        self,
        mesh: o3d.geometry.TriangleMesh,
        gt_surface_samples: np.ndarray,
    ):
        # sample the same ammout of points as in the gt_surface samples from the mesh
        n_samples = gt_surface_samples.shape[0]
        surface_samples = mesh.sample_points_uniformly(number_of_points=n_samples)
        surface_samples = np.asarray(surface_samples.points)

        latent_kd_tree = KDTree(surface_samples)
        latent_dist, _ = latent_kd_tree.query(gt_surface_samples)
        latent_chamfer = np.mean(np.square(latent_dist))

        gt_kd_tree = KDTree(gt_surface_samples)
        gt_dist, _ = gt_kd_tree.query(surface_samples)
        gt_chamfer = np.mean(np.square(gt_dist))

        return latent_chamfer + gt_chamfer
