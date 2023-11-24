import os
from pathlib import Path

import numpy as np

# import open3d as o3d
import torch
import trimesh
from skimage.measure import marching_cubes

from lib.models.deepsdf import DeepSDF


def reconstruct_training_data(
    model,
    checkpoint_path: str,
    resolution: int = 256,
    chunck_size: int = 500_000,  # based on RTX3090 / 16GB RAM (very rough estimate)
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = Path(checkpoint_path)
    save_path = str(path.parent.parent) + "/reconstructions"

    decoder = model.load_from_checkpoint(
        checkpoint_path
    ).cpu()  # in case settings suggest GPU
    decoder.eval()

    grid_vals = torch.arange(-1, 1, float(2 / resolution))
    grid = torch.meshgrid(grid_vals, grid_vals, grid_vals)

    xyz = torch.stack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).transpose(
        1, 0
    )
    n_chunks = (xyz.shape[0] // chunck_size) + 1
    #    a.element_size() * a.nelement()

    num_latents = decoder.lat_vecs.weight.shape[0]
    for i in range(num_latents):
        lat_vec = torch.Tensor([i]).int().repeat(xyz.shape[0])
        lat_vec = decoder.lat_vecs(lat_vec)

        # chunking
        decoder = decoder.to(device)
        xyz_chunks = xyz.unsqueeze(0).chunk(n_chunks, dim=1)
        lat_vec_chunks = lat_vec.unsqueeze(0).chunk(n_chunks, dim=1)
        sd_list = list()
        for _xyz, _lat_vec in zip(xyz_chunks, lat_vec_chunks):
            sd = (
                decoder.predict((_xyz.to(device), _lat_vec.to(device)))
                .squeeze()
                .cpu()
                .numpy()
            )
            sd_list.append(sd)
        sd = np.concatenate(sd_list)
        sd_r = sd.reshape(resolution, resolution, resolution)

        verts, faces, _, _ = marching_cubes(sd_r, level=0.0)

        x_max = np.array([1, 1, 1])
        x_min = np.array([-1, -1, -1])
        verts = verts * ((x_max - x_min) / (resolution)) + x_min

        # If it doesn't exist, create the directory
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Create a trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        # Save the mesh as an OBJ file
        mesh.export(f"{save_path}/{i}_{resolution}.obj")


for i in [64, 128, 256]:  # not enough VRAM for 512
    reconstruct_training_data(
        DeepSDF,
        "/home/korth/sketch2shape/logs/train/runs/2023-11-23_15-21-36/checkpoints/epoch_19999.ckpt",
        i,
    )
