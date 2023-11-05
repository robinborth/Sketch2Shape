import hydra
import lightning as L
from omegaconf import DictConfig

from lib.data.dataset import ShapeNetDataset
from lib.utils import create_logger, load_config

logger = create_logger("memory")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    logger.debug("==> loading config ...")
    L.seed_everything(cfg.seed)
    logger.debug("==> loading dataset ...")
    _ = ShapeNetDataset(cfg=cfg, stage="train")
    logger.debug("==> finishing memory profiling ...")


if __name__ == "__main__":
    train()
