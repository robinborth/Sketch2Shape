import hydra
from omegaconf import DictConfig
from optimize_latent import optimize_latent

from lib.utils import create_logger

log = create_logger("optimize_normals")


@hydra.main(version_base=None, config_path="../conf", config_name="optimize_normals")
def optimize(cfg: DictConfig) -> None:
    optimize_latent(cfg=cfg, log=log)


if __name__ == "__main__":
    optimize()
