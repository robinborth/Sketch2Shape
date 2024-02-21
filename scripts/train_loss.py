from typing import List

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from lib.eval.loss_tester import LossTester
from lib.utils.config import instantiate_callbacks, log_hyperparameters
from lib.utils.logger import create_logger

log = create_logger("train_loss")


@hydra.main(version_base=None, config_path="../conf", config_name="train_loss")
def train(cfg: DictConfig) -> None:
    log.info("==> loading config ...")
    L.seed_everything(cfg.seed)

    log.info("==> initializing logger ...")
    logger: Logger = hydra.utils.instantiate(cfg.logger)

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

    if cfg.get("train"):
        log.info("==> start training ...")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    if cfg.get("eval"):
        log.info("==> start testing ...")
        ckpt_path = trainer.checkpoint_callback.best_model_path  # type: ignore
        if ckpt_path == "":
            log.warning("==> best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        log.info(f"Best ckpt path: {ckpt_path}")

        log.info(f"==> initializing datamodule <{cfg.eval_data._target_}>")
        datamodule = hydra.utils.instantiate(cfg.eval_data)
        datamodule.setup("all")

        log.info(f"==> load model <{cfg.model._target_}>")
        tester = LossTester(ckpt_path, data_dir=cfg.data.data_dir)

        log.info(f"==> index datasets <{cfg.trainer._target_}>")
        cfg.trainer.max_epochs = 1
        trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
        trainer.validate(
            tester,
            dataloaders=[datamodule.val_dataloader()],
        )

        log.info("==> start testing ...")
        trainer.test(tester, dataloaders=datamodule.val_dataloader())


if __name__ == "__main__":
    train()
