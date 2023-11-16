import numpy as np
import open3d as o3d
import torch
from skimage.measure import marching_cubes
from pathlib import Path

from lib.models.deepsdf import DeepSDF


def reconstruct_training_data(
    model,
    checkpoint_path: str,
    resolution: int = 256,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = Path(checkpoint_path)
    save_path = str(path.parent) + "/reconstructions"

    decoder = model.load_from_checkpoint(checkpoint_path).to(device)
    decoder.eval()

    grid_vals = torch.arange(-1, 1, float(2 / resolution))
    grid = torch.meshgrid(grid_vals, grid_vals, grid_vals)

    xyz = (
        torch.stack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel()))
        .transpose(1, 0)
        .to(device)
    )

    num_latents = decoder.lat_vecs.weight.shape[0]
    for i in range(num_latents):
        lat_vec = torch.Tensor([i]).int().repeat(xyz.shape[0]).to(device)
        lat_vec = decoder.lat_vecs(lat_vec)
        sd = decoder((xyz.unsqueeze(0), lat_vec.unsqueeze(0)))
        sd_r = sd.reshape(resolution, resolution, resolution).detach().cpu().numpy()

        verts, faces, _, _ = marching_cubes(sd_r, level=0.0)

        x_max = np.array([1, 1, 1])
        x_min = np.array([-1, -1, -1])
        verts = verts * ((x_max - x_min) / (resolution)) + x_min
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d.io.write_triangle_mesh(f"{save_path}/{i}.obj", mesh)

reconstruct_training_data(
    DeepSDF,
    "/root/sketch2shape/logs/train/runs/2023-11-16_10-59-43/checkpoints/last.ckpt",
    256
)
