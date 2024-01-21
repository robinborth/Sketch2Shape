import os
from pathlib import Path

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


def create_video(video_dir: Path, obj_id: str, framerate: int = 30):
    image_folder = video_dir.parent / "wandb/latest-run/files/media/images"

    # rename so that ffmpeg can extract the correct order
    for video_frame_path in image_folder.glob("video_frame_*.png"):
        step = video_frame_path.name.split("_")[2].zfill(6)
        new_video_frame_path = video_frame_path.parent / f"video_frame_{step}.png"
        video_frame_path.rename(new_video_frame_path)

    image_folder_glob = (image_folder / "video_frame_*.png").as_posix()
    video_path = video_dir / f"{obj_id}.mp4"
    args: list[str] = [
        f"ffmpeg -framerate {framerate}",
        f'-pattern_type glob -i "{image_folder_glob}"',
        f"-pix_fmt yuv420p {video_path}",
    ]
    os.system(" ".join(args))
