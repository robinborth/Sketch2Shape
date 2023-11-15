# %%
import numpy as np
import open3d as o3d
import torch
from skimage.measure import marching_cubes

from lib.models.deepsdf import DeepSDF


def reconstruct_training_data(
    model,
    checkpoint_path: str,
    object_idx: int = 0,
    save_path: str = "",
    resolution: int = 64,
    visualize: bool = False,
):
    decoder = model.load_from_checkpoint(checkpoint_path)
    decoder.eval()

    grid_vals = torch.arange(-1, 1, float(2 / resolution))
    grid = torch.meshgrid(grid_vals, grid_vals, grid_vals)

    xyz = torch.stack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).transpose(
        1, 0
    )
    lat_vec = torch.zeros(xyz.shape[0]).int()
    lat_vec = decoder.lat_vecs(lat_vec)
    sd = decoder((xyz.unsqueeze(0), lat_vec.unsqueeze(0)))
    sd_r = sd.reshape(resolution, resolution, resolution).detach().numpy()

    verts, faces, _, _ = marching_cubes(sd_r, level=0.0)

    x_max = np.array([1, 1, 1])
    x_min = np.array([-1, -1, -1])
    verts = verts * ((x_max - x_min) / (resolution)) + x_min
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if not save_path:
        o3d.io.write_triangle_mesh(f"{save_path}/reconstruct.obj", mesh)
    if visualize:
        o3d.visualization.draw_plotly([mesh])


# reconstruct_training_data(ActualDeepSDF, "checkpoint/remote/loss=0.00.ckpt")
reconstruct_training_data(
    DeepSDF,
    "/Users/robinborth/Code/sketch2shape/logs/train/runs/2023-11-15_20-30-13/checkpoints/last.ckpt",
    0,
    visualize=True,
)

# %%
