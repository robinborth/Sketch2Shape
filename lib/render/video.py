import os
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from lib.models.deepsdf import DeepSDF


class VideoCamera:
    def __init__(
        self,
        # base settings
        deepsdf_ckpt_path: str = "deepsdf.ckpt",
        # video settings
        latent_dir: str = "/latent_dir",
        keystones: dict = {},
        # rendering settings
        n_render_steps: int = 100,
        clamp_sdf: float = 0.1,
        step_scale: float = 1.0,
        surface_eps: float = 1e-03,
        sphere_eps: float = 1e-01,
        normal_eps: float = 5e-03,
        mode: str = "grayscale",  # grayscale, normal
    ):
        # init deepsdf
        self.deepsdf = DeepSDF.load_from_checkpoint(
            deepsdf_ckpt_path,
            strict=True,
            # rendering settings
            n_render_steps=n_render_steps,
            clamp_sdf=clamp_sdf,
            step_scale=step_scale,
            surface_eps=surface_eps,
            sphere_eps=sphere_eps,
            normal_eps=normal_eps,
        )
        self.deepsdf.freeze()
        self.deepsdf.eval()

        self.latents = []
        for latent_path in tqdm(list(Path(latent_dir).iterdir())):
            latent = torch.load(latent_path).to(self.deepsdf.device)
            self.latents.append(latent)

        self.keystones = keystones
        self.mode = mode

    def create_camera(self, frame_idx: int):
        # TODO self.keystones
        azim = 40
        elev = -30
        self.deepsdf.create_camera(azim=azim, elev=elev)

    def create_frames(self) -> list:
        frames = []
        for frame_idx, latent in enumerate(self.latents):
            self.create_camera(frame_idx=frame_idx)
            image = self.deepsdf.capture_camera_frame(latent=latent, mode=self.mode)
            frame = (image.detach().cpu().numpy() * 255).astype(np.uint8)
            frames.append(Image.fromarray(frame))
        return frames

    def create_video(self, image_dir: Path, video_path: str, framerate: int = 30):
        image_folder_glob = (Path(image_dir) / "*.png").as_posix()
        args: list[str] = [
            f"ffmpeg -framerate {framerate}",
            f'-pattern_type glob -i "{image_folder_glob}"',
            f"-pix_fmt yuv420p {video_path}",
        ]
        os.system(" ".join(args))


def extract_frames(
    cfg: DictConfig,
):
    frames = []

    num_frames_to_extract = cfg.get("frames_to_extract")

    # Open the video file
    video = cv2.VideoCapture(cfg.get("input_video_path"))

    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the frame interval
    frame_interval = max(total_frames // num_frames_to_extract, 1)

    # Read frames at the specified interval
    frame_count = 0
    while video.isOpened() and frame_count < total_frames:
        # Read the current frame
        ret, frame = video.read()

        # If the frame was read successfully
        if ret:
            # Add the frame to the list of frames
            if frame_count % frame_interval == 0:
                frames.append(frame)
            frame_count += 1
        else:
            # Break the loop if the video is completed
            break

    # Release the video file
    video.release()

    print(f"Extracted {len(frames)} frames from video")

    # create folder if it does not exist (in python)
    os.makedirs(cfg.get("obj_dir"), exist_ok=True)

    # save the frames to disk
    for i, frame in enumerate(frames):
        cv2.imwrite(f"{cfg.get('obj_dir')}/{i:05}.png", frame)
