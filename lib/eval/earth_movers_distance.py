import numpy as np
import open3d as o3d
import torch
from numpy.random import choice
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torchmetrics import Metric


class EarthMoversDistance(Metric):
    def __init__(self, num_samples: int, seed: int = 123):
        super().__init__()
        self.num_samples = num_samples
        self.seed = seed
        self.add_state("emd", default=torch.tensor(0.0))
        self.add_state("total", default=torch.tensor(0))

    def sample_surface_samples(self, surface_samples):
        np.random.seed(self.seed)
        idx = choice(range(len(surface_samples)), self.num_samples, replace=False)
        return surface_samples[idx]

    def sample_mesh(self, mesh):
        np.random.seed(self.seed)
        return np.asarray(mesh.sample_points_uniformly(self.num_samples).points)

    def update(self, mesh: o3d.geometry.TriangleMesh, surface_samples: np.ndarray):
        gt_samples = self.sample_surface_samples(surface_samples)
        samples = self.sample_mesh(mesh)
        d = cdist(gt_samples, samples)
        assignment = linear_sum_assignment(d)
        emd = d[assignment].sum() / min(len(gt_samples), len(samples))
        self.emd += torch.tensor(emd).to(self.emd)
        self.total += 1

    def compute(self):
        return self.emd / self.total
