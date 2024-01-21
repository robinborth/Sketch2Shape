import glob
import os

import numpy as np


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


def create_video(
    run_folder: str,
    video_fname: str,
    framerate: int = 30,
):
    image_folder = run_folder + "/wandb/latest-run/files/media/images"
    mask = "video_frame_*.png"

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

    args: list[str] = [
        "ffmpeg",
        "-framerate",
        str(framerate),
        "-pattern_type",
        "glob",
        "-i",
        f"{image_folder}/{mask}",
        "-pix_fmt",
        "yuv420p",
        video_fname,
    ]
    os.system(" ".join(args))
