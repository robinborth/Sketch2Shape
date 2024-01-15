import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree


def compute_chamfer_distance(
    mesh: o3d.geometry.TriangleMesh,
    gt_surface_samples: np.ndarray,
):
    n_samples = gt_surface_samples.shape[0]
    surface_samples = o3d.sample_points.sample_points_uniformly(mesh, n_samples)

    latent_kd_tree = KDTree(surface_samples)
    latent_dist, _ = latent_kd_tree.query(gt_surface_samples)
    latent_chamfer = np.mean(np.square(latent_dist))

    gt_kd_tree = KDTree(gt_surface_samples)
    gt_dist, _ = gt_kd_tree.query(surface_samples)
    gt_chamfer = np.mean(np.square(gt_dist))

    return np.mean([latent_chamfer, gt_chamfer])
