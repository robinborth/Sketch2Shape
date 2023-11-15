from typing import List

import hydra
import lightning as L
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from lib.utils import create_logger, instantiate_loggers

log = create_logger("eval_siamese")


@hydra.main(version_base=None, config_path="../conf", config_name="eval_siamese")
def evaluate(cfg: DictConfig) -> None:
    log.info("==> checking checkpoint path ...")
    L.seed_everything(cfg.seed)
    assert cfg.ckpt_path

    log.info(f"==> initializing datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"==> initializing model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("==> initializing logger ...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"==> initializing trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    log.info("==> start testing ...")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    evaluate()
