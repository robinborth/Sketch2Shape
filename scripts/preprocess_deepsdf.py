import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import trimesh
from mesh_to_sdf import sample_sdf_near_surface
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="preprocess_deepsdf")
def preprocess(cfg: DictConfig) -> None:
    if not os.path.exists(cfg.data_dir):
        raise FileNotFoundError(f"Data Directory ${cfg.data_dir} does not exist.")

    L.seed_everything(cfg.seed)

    p = Path(cfg.data_dir)
    for file in p.glob("**/*.obj"):
        print(file)

        mesh = trimesh.load(file, force="mesh")
        mesh_sphere, points, sdf, offset, scale = sample_sdf_near_surface(mesh)

        data = np.column_stack((points, sdf))
        pos = data[:, 3] > 0

        save_path = file.parents[1]

        # save data
        np.savez(
            Path(save_path, "sdf_samples.npz"),
            pos=data[pos],
            neg=data[~pos],
        )

        # save scaling
        np.savez(Path(save_path, "scaling.npz"), offset=offset, scale=scale)

        # save unit sphere normalized obj file
        mesh_sphere.export(Path(save_path, "model_unit_sphere.obj"))


if __name__ == "__main__":
    preprocess()
