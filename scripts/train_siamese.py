from typing import List

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from lib.utils import create_logger, instantiate_callbacks, instantiate_loggers

log = create_logger("train_siamese")


@hydra.main(version_base=None, config_path="../conf", config_name="train_siamese")
def train(cfg: DictConfig) -> None:
    log.info("==> loading config ...")
    L.seed_everything(cfg.seed)

    log.info("==> initializing logger ...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"==> initializing datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"==> initializing model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("==> initializing callbacks ...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"==> initializing trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    if cfg.get("train"):
        log.info("==> start training ...")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    if cfg.get("test"):
        log.info("==> start testing ...")
        ckpt_path = trainer.checkpoint_callback.best_model_path  # type: ignore
        if ckpt_path == "":
            log.warning("==> best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    # if logger:
    #     logger[0].finish()  # type: ignore


if __name__ == "__main__":
    train()
