import glob
import os

import numpy as np
import torch


def get_translation(t):
    mat = np.identity(4)
    mat[2][3] += t
    return mat


def get_rotation_x(theta: float):
    _theta = np.deg2rad(np.array(theta, dtype=np.float32))
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(_theta), -np.sin(_theta), 0],
            [0, np.sin(_theta), np.cos(_theta), 0],
            [0, 0, 0, 1],
        ]
    )


def get_rotation_y(theta: float):
    _theta = np.deg2rad(np.array(theta, dtype=np.float32))
    return np.array(
        [
            [np.cos(_theta), 0, np.sin(_theta), 0],
            [0, 1, 0, 0],
            [-np.sin(_theta), 0, np.cos(_theta), 0],
            [0, 0, 0, 1],
        ]
    )


def get_rotation_z(theta: float):
    _theta = np.deg2rad(np.array(theta, dtype=np.float32))
    return np.array(
        [
            [np.cos(_theta), -np.sin(_theta), 0, 0],
            [np.sin(_theta), np.cos(_theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


# for Blinn-Phong model
class ReflectionProperty:
    def __init__(
        self,
        ambient: list = [0.3, 0.3, 0.3],
        diffuse: list = [0.3, 0.3, 0.3],
        specular: list = [0.0, 0.0, 0.0],
        shininess: int = 0,
    ):
        self.ambient = torch.tensor(ambient)
        self.diffuse = torch.tensor(diffuse)
        self.specular = torch.tensor(specular)
        self.shininess = torch.tensor(shininess)


class LightSource:
    def __init__(
        self,
        position: list = [0, 2, -2],
        ambient: list = [1, 1, 1],
        diffuse: list = [1, 1, 1],
        specular: list = [1, 1, 1],
    ):
        self.position = torch.tensor(position)
        # default values
        self.ambient = torch.tensor(ambient)
        self.diffuse = torch.tensor(diffuse)
        self.specular = torch.tensor(specular)


def create_video(
    run_folder: str,
    video_fname: str,
    framerate: int = 30,
):
    image_folder = run_folder + "/wandb/latest-run/files/media/images"
    mask = "default_image_*.png"
    # workaround since I was not able to get ffmpeg run
    for fname in glob.glob(image_folder + "/" + mask):
        paths = fname.split("/")
        last = paths[-1]
        last_split = last.split("_")
        last_split[2] = last_split[2].zfill(6)
        last = "_".join(last_split)
        paths[-1] = last
        new_fname = "/".join(paths)
        os.rename(fname, new_fname)

    os.system(
        f"ffmpeg -framerate {framerate} -pattern_type glob -i '{image_folder}/{mask}' -pix_fmt yuv420p {video_fname}"
    )
