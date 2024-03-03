from pathlib import Path

import hydra
from omegaconf import DictConfig
from optimize_latent import optimize_latent

from lib.render.utils import extract_frames_from_video
from lib.utils.logger import create_logger

log = create_logger("create_video")


@hydra.main(version_base=None, config_path="../conf", config_name="create_video")
def create_video(cfg: DictConfig) -> None:
    if cfg.preprocess_video:
        log.info("==> extract sketch frames ...")
        sketch_frames = extract_frames_from_video(
            video_path=cfg.input_video_path,
            skip_frames=cfg.skip_frames,
        )
        for frame_idx, frame in enumerate(sketch_frames):
            path = Path(cfg.obj_dir) / f"{frame_idx:05}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.save(path)

        log.info("==> optimize latent ...")
        optimize_latent(cfg=cfg, log=log)

    if cfg.capture_video:
        log.info("==> initializing video camera ...")
        video_camera = hydra.utils.instantiate(cfg.video_camera)
        video_camera.create_frames()

        log.info("==> capturing image video frames ...")
        for frame_idx, frame in enumerate(video_camera.images):
            path = Path(cfg.image_dir) / f"{frame_idx:05}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.save(path)
        log.info("==> save image video ...")
        video_camera.create_video(cfg.image_dir, cfg.output_video_path)

        log.info("==> capturing sketch image video frames ...")
        for frame_idx, frame in enumerate(video_camera.sketch_images):
            path = Path(cfg.sketch_image_dir) / f"{frame_idx:05}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.save(path)
        log.info("==> save sketch image video ...")
        video_camera.create_video(cfg.sketch_image_dir, cfg.input_output_video_path)


if __name__ == "__main__":
    create_video()
