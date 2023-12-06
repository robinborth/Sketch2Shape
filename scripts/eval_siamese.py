from typing import List

import hydra
import lightning as L
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights

from lib.eval.tester import SiameseTester
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

    log.info(f"==> load model <{cfg.model._target_}>")
    # model = Siamese.load_from_checkpoint(cfg.ckpt_path)
    # tester = SiameseTester(model=model.decoder)
    decoder = resnet18(ResNet18_Weights.IMAGENET1K_V1)
    decoder.fc = torch.nn.Identity()
    decoder.embedding_size = 512
    tester = SiameseTester(model=decoder)

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
