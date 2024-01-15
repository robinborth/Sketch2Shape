import glob
import json
import math
import os
from pathlib import Path
from typing import List

import hydra
import lightning as L
import numpy as np
import torch
import trimesh

# torch._dynamo.config.suppress_errors = True
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from lib.data.deepsdf_dataset import DeepSDFDataset
from lib.utils import create_logger, instantiate_callbacks, instantiate_loggers

log = create_logger("optimize_latent")


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="deepsdf_optimize_latent",
)
def optimize_latent(cfg: DictConfig) -> None:
    # if not os.path.exists(cfg.model.ckpt_path):
    #     raise ValueError("Please provide a checkpoint path to the DeepSDF Model")

    log.info("==> loading config ...")
    L.seed_everything(cfg.seed)

    log.info(f"==> initializing datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data,
    )

    log.info(f"==> initializing model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, ckpt_path=cfg.ckpt_path)

    log.info("==> initializing callbacks ...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("==> initializing logger ...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"==> initializing trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    log.info("==> start training ...")
    trainer.fit(model=model, datamodule=datamodule)

    model.get_obj()


if __name__ == "__main__":
    optimize_latent()
