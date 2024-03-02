from pathlib import Path

import hydra
from omegaconf import DictConfig

from lib.utils.logger import create_logger

log = create_logger("create_video")


@hydra.main(version_base=None, config_path="../conf", config_name="create_video")
def create_video(cfg: DictConfig) -> None:
    log.info("==> initializing video camera ...")
    video_camera = hydra.utils.instantiate(cfg.video_camera)
    video_frames = video_camera.create_frames()

    log.info("==> capturing video frames ...")
    for frame_idx, frame in enumerate(video_frames):
        path = Path(cfg.frame_dir) / f"{frame_idx:05}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.save(path)

    log.info("==> save video ...")
    video_camera.create_videos(
        image_dir=cfg.frame_dir,
        sketch_dir=cfg.obj_dir,
        video_path=cfg.output_video_path,
    )


if __name__ == "__main__":
    create_video()
