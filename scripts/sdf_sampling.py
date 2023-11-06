# %%
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from omegaconf import DictConfig

from lib.data.sketch import obj_path
from lib.utils import load_config


def create_sdf_samples_grid(
    obj_id: str,
    cfg: DictConfig,
    grid_size: int = 10,
    epsilon_dist: float = 0.1,
):
    # disable the warning
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # create mesh
    path = obj_path(obj_id, config=cfg)
    mesh = o3d.io.read_triangle_mesh(path)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # create grid
    min_bound = mesh.vertex.positions.min(0).numpy() - epsilon_dist
    max_bound = mesh.vertex.positions.max(0).numpy() + epsilon_dist
    xyz_range = np.linspace(min_bound, max_bound, num=grid_size)
    query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32)

    # retrieve the sdf values
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    signed_distance = scene.compute_signed_distance(query_points).numpy()
    return signed_distance


cfg = load_config()
obj_id = cfg.obj_ids[0]
sdf = create_sdf_samples_grid(obj_id=obj_id, cfg=cfg, grid_size=10)
plt.imshow(sdf[:, :, 0])
