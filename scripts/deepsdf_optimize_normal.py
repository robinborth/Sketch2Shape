from typing import List

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from lib.utils import create_logger, instantiate_callbacks, instantiate_loggers

log = create_logger("optimize_normal")


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="deepsdf_optimize_normal",
)
def optimize(cfg: DictConfig) -> None:
    log.info("==> loading config ...")
    L.seed_everything(cfg.seed)

    log.info(f"==> initializing datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"==> initializing model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("==> initializing callbacks ...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("==> initializing logger ...")
    logger: WandbLogger = instantiate_loggers(cfg.get("logger"))
    if logger is not None:
        logger.watch(model, log="all")

    log.info(f"==> initializing trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    log.info("==> start training ...")
    trainer.fit(model=model, datamodule=datamodule)

    if cfg.save_obj:
        log.info("==> save object ...")
        mesh = model.to_mesh()
        mesh.export(f"{cfg.save_obj_path}/{trainer.max_epochs}-test.obj")


if __name__ == "__main__":
    optimize()
