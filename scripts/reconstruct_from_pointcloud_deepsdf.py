import glob
import json
import os
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import trimesh
from omegaconf import DictConfig, OmegaConf
import torch
from skimage.measure import marching_cubes

from lib.evaluate import compute_chamfer_distance
from lib.generate import reconstruct_training_data
from lib.models.deepsdf import DeepSDF
import math


def load_run_config(path: str):
    return OmegaConf.load(Path(path, ".hydra/config.yaml"))


def load_checkpoint_path(path: str):
    paths = glob.glob(path + "/**/last.ckpt", recursive=True)
    if paths:
        return paths[0]
    raise ValueError(f"Could not find a last.ckpt file in {path}")


def get_id_from_path(rec_path):
    return rec_path.split("/")[-1].split("_")[0]

def load_pointcloud(path):
    path = "/root/sketch2shape/data/SurfaceSamples/overfit_1/787a4db5b3452fc357a847db7547c1f3.ply"
    return trimesh.load(path).vertices


@hydra.main(version_base=None, config_path="../conf", config_name="reconstruct_pointcloud_deepsdf")
def reconstruct_pointcloud(cfg: DictConfig) -> None:
    if not os.path.exists(cfg.log_path):
        raise ValueError("Please provide a log path")

    L.seed_everything(cfg.seed)
    run_config = load_run_config(cfg.log_path)

    ckpt_path = (
        cfg.ckpt_path
        if cfg.ckpt_path is not None
        else load_checkpoint_path(cfg.log_path)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    deepsdf = DeepSDF.load_from_checkpoint(ckpt_path).to(device)
    deepsdf.eval()

    points = load_pointcloud("") #cfg.pointcloud_path)
    points = torch.Tensor(points).to(device)

    lat_vecs = torch.nn.Embedding(1, run_config.model.latent_vector_size).to(device)
    std_lat_vec = 1.0 / math.sqrt(run_config.model.latent_vector_size)
    torch.nn.init.normal_(lat_vecs.weight.data, 0.0, std_lat_vec)

    optim = torch.optim.Adam(lat_vecs.parameters(), lr=1e-3)
    l1 = torch.nn.L1Loss(reduction='mean')

    idx = torch.zeros(points.shape[0]).int().to(device)
    y = torch.zeros(points.shape[0]).to(device)

    for i in range(cfg.n_steps):
        optim.zero_grad()
        
        lat_vec = lat_vecs(idx)

        y_hat = deepsdf((points.unsqueeze(0), lat_vec.unsqueeze(0))).flatten()

        loss = l1(y_hat, y)
        print(loss)
        loss.backward()
        optim.step()

    # workaround
    # hparams
    resolution =  64
    chunck_size = 500_000
    decoder = deepsdf
    lat_vecs.cpu()
    # hparams
    grid_vals = torch.arange(-1, 1, float(2 / resolution))
    grid = torch.meshgrid(grid_vals, grid_vals, grid_vals)

    xyz = torch.stack(
        (grid[0].ravel(), grid[1].ravel(), grid[2].ravel())
    ).transpose(1, 0)

    del grid, grid_vals

    # based on rough trial and error
    n_chunks = (xyz.shape[0] // chunck_size) + 1
    #    a.element_size() * a.nelement()

    num_latents = lat_vecs.weight.shape[0]
    for i in range(num_latents):
        lat_vec = torch.Tensor([i]).int().repeat(xyz.shape[0])
        lat_vec = lat_vecs(lat_vec)

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
        decoder = decoder.cpu()
        sd = np.concatenate(sd_list)
        sd_r = sd.reshape(resolution, resolution, resolution)

        print(sd_r)
        print(sd_r.min())

        verts, faces, _, _ = marching_cubes(sd_r, level=0.0)

        x_max = np.array([1, 1, 1])
        x_min = np.array([-1, -1, -1])
        verts = verts * ((x_max - x_min) / (resolution)) + x_min

        # Create a trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        # remove objects outside unit sphere
        # mesh = remove_faces_outside_sphere(mesh)

        path_obj = f"/root/sketch2shape/test.obj"
        # Save the mesh as an OBJ file
        mesh.export(path_obj)


if __name__ == "__main__":
    reconstruct_pointcloud()
