import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig

from lib.data.datamodule import SDFDataModule
from lib.utils import create_logger

logger = create_logger("train_deepsdf")


@hydra.main(version_base=None, config_path="../conf", config_name="config_sdf")
def train(cfg: DictConfig) -> None:
    logger.debug("==> loading config ...")
    L.seed_everything(cfg.seed)

    logger.debug("==> initializing datamodule ...")
    datamodule = SDFDataModule(cfg=cfg)

    logger.debug("==> initializing model ...")
    model = instantiate(cfg.model, cfg)

    logger.debug("==> initializing callbacks ...")
    callbacks = [instantiate(callback) for callback in cfg.callbacks.values()]

    logger.debug("==> initializing logger ...")
    wandb_logger = instantiate(cfg.logging)
    wandb_logger.watch(model, log="all", log_freq=10, log_graph=False)

    logger.debug("==> initializing trainer ...")
    trainer = instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=wandb_logger,
    )

    logger.debug("==> start training ...")
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train()
