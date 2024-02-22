import hydra
import lightning as L
from lightning import LightningDataModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from lib.eval.loss_tester import LossTester
from lib.utils.logger import create_logger

log = create_logger("eval_loss")


@hydra.main(version_base=None, config_path="../conf", config_name="eval_loss")
def evaluate(cfg: DictConfig) -> None:
    log.info("==> checking checkpoint path ...")
    L.seed_everything(cfg.seed)
    assert cfg.loss_ckpt_path

    log.info("==> initializing logger ...")
    logger: Logger = hydra.utils.instantiate(cfg.logger)

    log.info(f"==> initializing datamodule <{cfg.eval_data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.eval_data)
    datamodule.setup("validate")

    log.info("==> load tester ...")
    tester = LossTester(loss_ckpt_path=cfg.loss_ckpt_path, data_dir=cfg.data.data_dir)

    log.info(f"==> index datasets <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    trainer.validate(
        tester,
        dataloaders=[datamodule.val_dataloader()],
    )

    log.info("==> start testing ...")
    trainer.test(tester, dataloaders=datamodule.val_dataloader())


if __name__ == "__main__":
    evaluate()
