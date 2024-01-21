from pathlib import Path

import hydra
import lightning as L
import open3d as o3d
import pandas as pd
import wandb
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from lib.data.metainfo import MetaInfo
from lib.utils import create_logger, instantiate_callbacks, log_hyperparameters

log = create_logger("optimize_deepsdf")


@hydra.main(version_base=None, config_path="../conf", config_name="optimize_deepsdf")
def optimize(cfg: DictConfig) -> None:
    metrics = []

    log.info("==> loading config ...")
    assert cfg.data.batch_size == 1  # make sure that the batch_size is 1
    L.seed_everything(cfg.seed)

    metainfo = MetaInfo(data_dir=cfg.data.data_dir, split=cfg.split)
    if obj_ids := cfg.get("obj_ids"):  # specific obj_ids are selected
        log.info(f"==> selecting specified obj_ids ({len(obj_ids)}) ...>")
    if obj_ids is None:  # if there are specific obj_ids we use all from a split
        obj_ids = metainfo.obj_ids
        log.info(f"==> selecting obj_ids ({cfg.split}) ...>")

    if cfg.split != "train" and cfg.prior_idx:
        log.info("WARNING! Only select prior_idx for split=train ...>")
        log.info("==> Set prior_idx=False ...>")
        cfg.prior_idx = False

    if not cfg.eval and cfg.save_mesh:
        log.info("WARNING! The mesh is only created in the eval loop ...>")
        log.info("==> Set eval=True ...>")
        cfg.eval = True

    for obj_id in obj_ids:
        log.info(f"==> optimize {obj_id=} ...")
        cfg.data.obj_id = obj_id

        log.info(f"==> initializing datamodule <{cfg.data._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

        log.info(f"==> initializing model <{cfg.model._target_}>")
        if cfg.prior_idx:  # set the prior_idx for the trained shapes
            cfg.model.prior_idx = metainfo.obj_id_to_label(obj_id=obj_id)
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        log.info("==> initializing callbacks ...")
        callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

        log.info("==> initializing logger ...")
        logger: Logger = hydra.utils.instantiate(cfg.logger)

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
    df.to_csv(cfg.paths.metrics_path, index=False)

    log.info("==> save mean metrics ...")
    mean_metric = df.loc[:, df.columns != "obj_id"].mean()
    mean_metric.to_csv(cfg.paths.mean_metrics_path)


if __name__ == "__main__":
    optimize()
