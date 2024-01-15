from typing import List

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from lib.data.metainfo import MetaInfo
from lib.utils import (
    create_logger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
)

log = create_logger("optimize_deepsdf")


@hydra.main(version_base=None, config_path="../conf", config_name="optimize_deepsdf")
def optimize(cfg: DictConfig) -> None:
    log.info("==> loading config ...")
    L.seed_everything(cfg.seed)

    log.info(f"==> initializing datamodule <{cfg.data._target_}>")
    metainfo = MetaInfo(data_dir=cfg.data.data_dir)
    obj_id = metainfo.obj_ids[0]
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, obj_id=obj_id)

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

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    if logger:
        log.info("==> logging hyperparameters ...")
        log_hyperparameters(object_dict)

    log.info("==> start training ...")
    trainer.fit(model=model, datamodule=datamodule)

    if cfg.save_obj:
        log.info("==> save object ...")
        mesh = model.to_mesh(resolution=128)
        mesh.export(f"{cfg.save_obj_path}/{trainer.max_epochs}-test.obj")


if __name__ == "__main__":
    optimize()
