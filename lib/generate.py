import glob
import os
from pathlib import Path

import numpy as np
import torch
import trimesh
from skimage.measure import marching_cubes


def remove_faces_outside_sphere(mesh: trimesh.Trimesh, sphere=1.03):
    vertex_norms = np.linalg.norm(mesh.vertices, axis=1)
    indices_to_remove = np.where(vertex_norms > sphere)[0]

    mask = np.zeros(mesh.faces.shape[0], bool)
    for i, face in enumerate(mesh.faces):
        for vertex in face:
            if vertex in indices_to_remove:
                mask[i] = 1
                break

    mesh.faces = mesh.faces[~mask]
    mesh.remove_unreferenced_vertices()
    return mesh


def reconstruct_training_data(
    model,
    checkpoint_path: str,
    idx2shape: dict,
    resolution_list: list = [64],
    chunck_size: int = 500_000,  # based on 2080TI / 16GB RAM (very rough estimate)
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = Path(checkpoint_path)
    save_path = str(path.parent.parent) + "/reconstructions"
    # If it doesn't exist, create the directory; otherwise, skip computation ()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        print("Reconstruction folder already exists, skipping reconstruction")
        return glob.glob(save_path + "/**/*.obj", recursive=True)
    list_obj_paths = list()

    decoder = model.load_from_checkpoint(
        checkpoint_path
    ).cpu()  # in case settings suggest GPU

    decoder.eval()
    decoder.lat_vecs.weight.requires_grad = False

    for resolution in resolution_list:
        grid_vals = torch.arange(-1, 1, float(2 / resolution))
        grid = torch.meshgrid(grid_vals, grid_vals, grid_vals)

        xyz = torch.stack(
            (grid[0].ravel(), grid[1].ravel(), grid[2].ravel())
        ).transpose(1, 0)

        del grid, grid_vals

        # based on rough trial and error
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
            decoder = decoder.cpu()
            sd = np.concatenate(sd_list)
            sd_r = sd.reshape(resolution, resolution, resolution)

            verts, faces, _, _ = marching_cubes(sd_r, level=0.0)

            x_max = np.array([1, 1, 1])
            x_min = np.array([-1, -1, -1])
            verts = verts * ((x_max - x_min) / (resolution)) + x_min

            # Create a trimesh object
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)

            # remove objects outside unit sphere
            mesh = remove_faces_outside_sphere(mesh)

            path_obj = f"{save_path}/{idx2shape[i]}_{resolution}.obj"
            # Save the mesh as an OBJ file
            mesh.export(path_obj)

            del (
                sd,
                sd_r,
                lat_vec,
                lat_vec_chunks,
                sd_list,
                xyz_chunks,
                _xyz,
                _lat_vec,
                verts,
                faces,
            )
            import gc

            gc.collect()

            list_obj_paths.append(path_obj)
    return list_obj_paths


def traverse_latent_space(
    model,
    checkpoint_path: str,
    idx2shape: dict,
    idx_one: int = 0,
    idx_two: int = 1,
    steps: int = 7,
    resolution_list: list = [256],
    chunck_size: int = 500_000,  # based on RTX3090 / 16GB RAM (very rough estimate)
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = Path(checkpoint_path)
    save_path = str(path.parent.parent) + "/traversal"
    # If it doesn't exist, create the directory; otherwise, skip computation ()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    decoder = model.load_from_checkpoint(
        checkpoint_path
    ).cpu()  # in case settings suggest GPU

    decoder.eval()
    decoder.lat_vecs.weight.requires_grad = False

    for resolution in resolution_list:
        grid_vals = torch.arange(-1, 1, float(2 / resolution))
        grid = torch.meshgrid(grid_vals, grid_vals, grid_vals)

        xyz = torch.stack(
            (grid[0].ravel(), grid[1].ravel(), grid[2].ravel())
        ).transpose(1, 0)

        del grid, grid_vals

        # based on rough trial and error
        n_chunks = (xyz.shape[0] // chunck_size) + 1
        #    a.element_size() * a.nelement()

        # select two latents to traverse
        lat_vec_one = decoder.lat_vecs(torch.Tensor([idx_one]).int())
        lat_vec_two = decoder.lat_vecs(torch.Tensor([idx_two]).int())

        for t in np.linspace(0, 1, steps):
            lat_vec = t * lat_vec_one + (1 - t) * lat_vec_two
            lat_vec = lat_vec.repeat(xyz.shape[0], 1)

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

            verts, faces, _, _ = marching_cubes(sd_r, level=0.0)

            x_max = np.array([1, 1, 1])
            x_min = np.array([-1, -1, -1])
            verts = verts * ((x_max - x_min) / (resolution)) + x_min

            # Create a trimesh object
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)

            # remove objects outside unit sphere
            mesh = remove_faces_outside_sphere(mesh)

            path_obj = f"{save_path}/t_{t}.obj"
            # Save the mesh as an OBJ file
            mesh.export(path_obj)

            del (
                sd,
                sd_r,
                lat_vec,
                lat_vec_chunks,
                sd_list,
                xyz_chunks,
                _xyz,
                _lat_vec,
                verts,
                faces,
            )
            import gc

            gc.collect()
