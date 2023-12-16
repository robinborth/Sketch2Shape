from typing import List

import hydra
import lightning as L
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from lib.eval.tester import SiameseTester
from lib.models.decoder import EvalCLIP, EvalResNet18
from lib.models.siamese import Siamese
from lib.utils import create_logger, instantiate_loggers

log = create_logger("eval_siamese")


@hydra.main(version_base=None, config_path="../conf", config_name="eval_siamese")
def evaluate(cfg: DictConfig) -> None:
    log.info("==> checking checkpoint path ...")
    L.seed_everything(cfg.seed)
    # assert cfg.ckpt_path

    log.info("==> initializing logger ...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"==> initializing datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("all")

    if cfg.ckpt_path == "resnet18":
        model = EvalResNet18()
    elif cfg.ckpt_path == "clip":
        model = EvalCLIP()
    else:
        model = Siamese.load_from_checkpoint(cfg.ckpt_path).decoder
    log.info(f"==> load model <{model}>")
    model.metainfo = datamodule.metainfo
    tester = SiameseTester(model=model)

    log.info(f"==> index datasets <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    trainer.validate(
        tester,
        dataloaders=[
            datamodule.train_dataloader(),
            datamodule.val_dataloader(),
        ],
    )

    log.info("==> start testing ...")
    trainer.test(tester, dataloaders=datamodule.val_dataloader())


if __name__ == "__main__":
    evaluate()
