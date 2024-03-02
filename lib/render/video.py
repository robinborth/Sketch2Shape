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
        rotate: bool = False,
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
        for latent_path in tqdm(list(sorted(Path(latent_dir).iterdir()))):
            latent = torch.load(latent_path).to(self.deepsdf.device)
            self.latents.append(latent)

        self.rotate = rotate
        if self.rotate:
            self.keystones = np.arange(0, 360, 20)
        self.mode = mode

    def create_camera(self, frame_idx: int):
        azim = self.keystones[frame_idx % len(self.keystones)] if self.rotate else 40
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

    def create_videos(
        self,
        image_dir: Path,
        sketch_dir: Path,
        video_path: str,
        framerate: int = 10,
        side_by_side: bool = True,
    ):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = framerate

        # load images into array
        images = []
        for image_path in sorted(Path(image_dir).iterdir()):
            images.append(cv2.imread(image_path.as_posix()))
        # load sketches into array
        sketches = []
        for sketch_path in sorted(Path(sketch_dir).iterdir()):
            sketches.append(cv2.imread(sketch_path.as_posix()))

        res_frames = images[0].shape[:2][::-1]
        print(f"Frames resolution: {res_frames}")
        res_sketch = sketches[0].shape[:2][::-1]
        print(f"Sketch resolution: {res_sketch}")

        image_video_writer = cv2.VideoWriter(
            video_path + "image.mp4", fourcc, fps, res_frames
        )
        sketch_video_writer = cv2.VideoWriter(
            video_path + "sketch.mp4", fourcc, fps, res_sketch
        )

        for i in range(len(images)):
            image_video_writer.write(images[i])
            sketch_video_writer.write(sketches[i])

        image_video_writer.release()
        sketch_video_writer.release()

        # not working yet
        if side_by_side:
            res = min(res_frames[0], res_sketch[0])
            res_side_by_side = (res * 2, res)
            print(f"Side by side resolution: {res_side_by_side}")

            side_by_side_video_writer = cv2.VideoWriter(
                video_path + "side_by_side.mp4", fourcc, fps, res_side_by_side
            )

            for i in range(len(images)):
                image_resized = cv2.resize(images[i], (res, res))
                sketch_resized = cv2.resize(sketches[i], (res, res))
                side_by_side_frame = np.concatenate(
                    [sketch_resized, image_resized], axis=1
                )
                side_by_side_video_writer.write(side_by_side_frame)

            side_by_side_video_writer.release()


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
