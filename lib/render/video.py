import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm

from lib.models.deepsdf import DeepSDF


class VideoCamera:
    def __init__(
        self,
        # base settings
        deepsdf_ckpt_path: str = "deepsdf.ckpt",
        # video settings
        latent_dir: str = "/latent_dir",
        sketch_dir: str = "/sketch_dir",
        # rendering settings
        n_render_steps: int = 100,
        clamp_sdf: float = 0.1,
        step_scale: float = 1.0,
        surface_eps: float = 1e-03,
        sphere_eps: float = 1e-01,
        normal_eps: float = 5e-03,
        mode: str = "grayscale",  # grayscale, normal
        # rotate settings
        azim: float = 40,
        elev: float = -30,
        rotate_at_frame: list[str | int] = [100, "end"],
        rotation_step_size: int = 5,
        rotation_wait_frames: int = 5,
    ):
        self.transform = v2.Compose(
            [v2.ToImage(), v2.Resize((256, 256)), v2.ToPILImage()]
        )

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

        self.mode = mode

        self.latents = []
        for latent_path in tqdm(list(sorted(Path(latent_dir).iterdir()))):
            latent = torch.load(latent_path).to(self.deepsdf.device)
            self.latents.append(latent)

        self.sketches = []
        for sketch_path in tqdm(list(sorted(Path(sketch_dir).iterdir()))):
            sketch = np.array(Image.open(sketch_path))
            self.sketches.append(self.transform(sketch))

        # initilize the keystones
        self.keystones = {}
        idx = 0
        key_idx = 0
        num_rotations = 0
        while idx < len(self.latents):
            # get the next rotate_frame_idx
            rotate_frame_idx = rotate_at_frame[num_rotations]
            if rotate_frame_idx == "end":
                rotate_frame_idx = len(self.latents) - 1

            self.keystones[key_idx] = {
                "azim": azim,
                "elev": elev,
                "latent": self.latents[idx],
                "sketch": self.sketches[idx],
            }
            if idx == rotate_frame_idx:
                for d_azim in np.arange(0, 360, rotation_step_size):
                    self.keystones[key_idx] = {
                        "azim": azim + d_azim,
                        "elev": elev,
                        "latent": self.latents[idx],
                        "sketch": self.sketches[idx],
                    }
                    key_idx += 1
                for _ in range(rotation_wait_frames):
                    self.keystones[key_idx] = {
                        "azim": azim + d_azim,
                        "elev": elev,
                        "latent": self.latents[idx],
                        "sketch": self.sketches[idx],
                    }
                    key_idx += 1
                num_rotations += 1

            key_idx += 1
            idx += 1

            if num_rotations >= len(rotate_at_frame):
                num_rotations = len(rotate_at_frame) - 1

    def create_frames(self):
        images = []
        sketch_images = []
        for f in tqdm(self.keystones.values()):
            self.deepsdf.create_camera(azim=f["azim"], elev=f["elev"])
            image = self.deepsdf.capture_camera_frame(f["latent"], mode=self.mode)
            # single view just the rendered frame
            frame = self.transform(image.detach().cpu().numpy())  # (256, 256, 3)
            images.append(frame)
            # left sketch right rendered frame
            frame = np.concatenate([np.array(f["sketch"]), np.array(frame)], axis=1)
            sketch_images.append(Image.fromarray(frame))
        self.images = images
        self.sketch_images = sketch_images

    def create_video(self, image_dir: Path, video_path: str, framerate: int = 30):
        image_folder_glob = (Path(image_dir) / "*.png").as_posix()
        args: list[str] = [
            f"ffmpeg -framerate {framerate}",
            f'-pattern_type glob -i "{image_folder_glob}"',
            f"-pix_fmt yuv420p {video_path}",
            "-y",
        ]
        os.system(" ".join(args))
