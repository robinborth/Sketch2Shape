import glob
import re
from pathlib import Path
from typing import List

import hydra
import lightning as L
import numpy as np
import torch
import wandb

# torch._dynamo.config.suppress_errors = True
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from lib.utils import create_logger, instantiate_callbacks, instantiate_loggers

log = create_logger("optimize_latent")


def extract_ckpt_and_epoch(path):
    ckpt_paths = glob.glob(path + "/*9.ckpt")
    ckpt_epochs = [
        int(re.findall("\d+", ckpt.split("/")[-1])[0]) for ckpt in ckpt_paths
    ]
    return list(  # :D
        reversed(
            sorted(
                [
                    (ckpt_epoch, ckpt_path)
                    for ckpt_epoch, ckpt_path in zip(ckpt_epochs, ckpt_paths)
                ]
            )
        )
    )


# TODO
# [ ] enable both training and validation optimization at the same time
# [ ] document what exactly this loop does
@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="eval_deepsdf",
)
def eval(cfg: DictConfig) -> None:
    log.info("==> loading config ...")
    L.seed_everything(cfg.seed)

    log.info(
        f"==> initializing datamodule <{cfg.data_train._target_}> with data {cfg.data_train.experiment}"
    )
    datamodule_train: LightningDataModule = hydra.utils.instantiate(
        cfg.data_train,
    )
    log.info(
        f"==> initializing datamodule <{cfg.data_val._target_}> with data {cfg.data_val.experiment}"
    )
    datamodule_val: LightningDataModule = hydra.utils.instantiate(
        cfg.data_val,
    )

    ckpts = extract_ckpt_and_epoch(cfg.ckpt_folder)

    ckpts = ckpts[:: cfg.eval_every_n_ckpt]

    log.info("==> initializing callbacks ...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # # TODO clean debug
    # epoch, ckpt = ckpts[0]
    # log.info(
    #     f"==> initializing model <{cfg.model._target_}> from checkpoint at epoch: {epoch}"
    # )
    # model: LightningModule = hydra.utils.instantiate(cfg.model, ckpt_path=ckpt)

    # from lightning.pytorch.profilers import AdvancedProfiler

    # profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")

    # log.info(f"==> initializing trainer <{cfg.trainer._target_}>")
    # trainer: Trainer = hydra.utils.instantiate(
    #     cfg.trainer,
    #     callbacks=callbacks,
    #     # logger=logger,
    #     num_sanity_val_steps=0,
    # )

    # log.info("==> start training ...")
    # trainer.fit(model=model, datamodule=datamodule)

    for epoch, ckpt in ckpts:  # different ckpt
        for name, datamodule in zip(
            ["train", "val"], [datamodule_train, datamodule_val]
        ):
            log.info(
                f"==> initializing model <{cfg.model._target_}> from checkpoint at epoch: {epoch}"
            )
            model: LightningModule = hydra.utils.instantiate(cfg.model, ckpt_path=ckpt)

            log.info(f"==> using {name} datamodule")

            cfg["logger"]["wandb"]["name"] = f"{name}_epoch_{epoch}"
            log.info("==> initializing logger ...")
            logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

            log.info(f"==> initializing trainer <{cfg.trainer._target_}>")
            trainer: Trainer = hydra.utils.instantiate(
                cfg.trainer,
                callbacks=callbacks,
                logger=logger,
                num_sanity_val_steps=0,
            )

            log.info("==> start training ...")
            trainer.fit(model=model, datamodule=datamodule)

            wandb.finish()

        if cfg.save_outcomes:
            np.save(
                f"{cfg.paths.output_dir}/{name}-{epoch}-idx2chamfer", model.idx2chamfer
            )
            np.save(f"{cfg.paths.output_dir}/{name}-{epoch}-shape2idx", model.shape2idx)
            torch.save(
                model.lat_vecs, f"{cfg.paths.output_dir}/{name}-{epoch}-lat_vecs.pt"
            )


if __name__ == "__main__":
    eval()
