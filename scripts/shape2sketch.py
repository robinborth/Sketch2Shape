import hydra
from omegaconf import DictConfig

from lib.utils import create_logger

logger = create_logger("shape2sketch")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.debug("==> extracting images...")
    print(cfg)


if __name__ == "__main__":
    main()
