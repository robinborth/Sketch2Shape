import hydra
from omegaconf import DictConfig

from lib.utils.logger import create_logger

log = create_logger("create_video")


@hydra.main(version_base=None, config_path="../conf", config_name="create_video")
def optimize(cfg: DictConfig) -> None:
    video_camera = hydra.utils.instantiate(cfg.video_camera)
    video_frames = video_camera.create()

    assert cfg.video_dir
    for frame in video_frames:
        print("save frame")


if __name__ == "__main__":
    optimize()
