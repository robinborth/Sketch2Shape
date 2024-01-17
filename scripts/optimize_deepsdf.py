from pathlib import Path
from typing import List

import hydra
import lightning as L
import open3d as o3d
import pandas as pd
import wandb
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
    metrics = []

    log.info("==> loading config ...")
    assert cfg.data.batch_size == 1  # make sure that the batch_size is 1
    L.seed_everything(cfg.seed)

    log.info(f"==> initializing datamodule <{cfg.data._target_}>")
    metainfo = MetaInfo(data_dir=cfg.data.data_dir, split=cfg.split)
    for obj_id in metainfo.obj_ids:
        log.info(f"==> optimize {obj_id=} ...")
        cfg.data.obj_id = obj_id
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

        log.info(f"==> initializing model <{cfg.model._target_}>")
        cfg.model.prior_idx = metainfo.obj_id_to_label(obj_id=obj_id)
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        log.info("==> initializing callbacks ...")
        callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

        log.info("==> initializing logger ...")
        logger: WandbLogger = instantiate_loggers(cfg.get("logger"))

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

        if cfg.train:
            log.info("==> start training ...")
            trainer.fit(model=model, datamodule=datamodule)

        if cfg.eval:
            log.info("==> start evalution ...")
            metric = trainer.test(model=model, datamodule=datamodule)[0]
            metric["obj_id"] = obj_id  # type: ignore
            metrics.append(metric)

        if cfg.save_mesh and (mesh := model.mesh):
            log.info("==> save mesh ...")
            path = Path(cfg.paths.mesh_dir, f"{obj_id}.obj")
            path.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_triangle_mesh(
                path.as_posix(),
                mesh=mesh,
                write_triangle_uvs=False,
            )

        # finish the wandb run in order to track all the optimizations seperate
        wandb.finish()

    log.info("==> save metrics ...")
    df = pd.DataFrame(metrics)
    mean_metric = df.loc[:, df.columns != "obj_id"].mean()
    mean_metric["obj_id"] = "mean_metric"
    df = pd.concat([df, mean_metric.to_frame().T], ignore_index=True)
    df.to_csv(cfg.paths.metrics_path, index=False)


if __name__ == "__main__":
    optimize()
