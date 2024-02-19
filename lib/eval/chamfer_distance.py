import numpy as np
import open3d as o3d
import torch
from numpy.random import choice
from sklearn.neighbors import KDTree
from torchmetrics import Metric


class ChamferDistance(Metric):
    def __init__(self, num_samples: int, seed: int = 123):
        super().__init__()
        self.num_samples = num_samples
        self.seed = seed
        self.add_state("cd", default=torch.tensor(0.0))
        self.add_state("total", default=torch.tensor(0))

    def sample_surface_samples(self, surface_samples):
        np.random.seed(self.seed)
        idx = choice(range(len(surface_samples)), self.num_samples, replace=False)
        return surface_samples[idx]

    def sample_mesh(self, mesh):
        np.random.seed(self.seed)
        return np.asarray(mesh.sample_points_uniformly(self.num_samples).points)

    def update(self, mesh: o3d.geometry.TriangleMesh, surface_samples: np.ndarray):
        # sample the same ammout of points as in the gt_surface samples from the mesh
        gt_samples = self.sample_surface_samples(surface_samples)
        samples = self.sample_mesh(mesh)

        latent_kd_tree = KDTree(samples)
        latent_dist, _ = latent_kd_tree.query(gt_samples)
        latent_chamfer = np.mean(np.square(latent_dist))

        gt_kd_tree = KDTree(gt_samples)
        gt_dist, _ = gt_kd_tree.query(samples)
        gt_chamfer = np.mean(np.square(gt_dist))

        cd = latent_chamfer + gt_chamfer
        self.cd += torch.tensor(cd).to(self.cd)
        self.total += 1

    def compute(self) -> float:
        return (self.cd / self.total).item()
