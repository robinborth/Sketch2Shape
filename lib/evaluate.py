import numpy as np
import trimesh
from sklearn.neighbors import KDTree


def compute_chamfer_distance(gt, rec_mesh, n_samples=30000):
    rec_samples = trimesh.sample.sample_surface(rec_mesh, n_samples)[0]
    # rec_samples = rec_samples / norm_file["scale"] - norm_file["offset"]
    gt_samples = gt.vertices

    rec_kd_tree = KDTree(rec_samples)
    dist, _ = rec_kd_tree.query(gt_samples)
    gt_to_rec_chamfer = np.mean(np.square(dist))

    gt_kd_tree = KDTree(gt_samples)
    dist2, _ = gt_kd_tree.query(rec_samples)
    rec_to_gt_chamfer = np.mean(np.square(dist2))

    return gt_to_rec_chamfer + rec_to_gt_chamfer
