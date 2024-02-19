import re
from logging import Logger
from pathlib import Path

import hydra
import lightning as L
import open3d as o3d
import pandas as pd
import wandb
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger as LightningLogger
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from lib.data.metainfo import MetaInfo
from lib.eval.chamfer_distance import ChamferDistance
from lib.eval.earth_movers_distance import EarthMoversDistance
from lib.render.utils import create_video
from lib.utils.config import instantiate_callbacks, log_hyperparameters


def optimize_latent(cfg: DictConfig, log: Logger) -> None:
    log.info("==> loading config ...")
    assert cfg.data.batch_size == 1  # make sure that the batch_size is 1
    L.seed_everything(cfg.seed)

    metainfo = MetaInfo(data_dir=cfg.data.data_dir, split=cfg.split)
    if obj_ids := cfg.get("obj_ids"):  # specific obj_ids are selected
        log.info(f"==> selecting specified obj_ids ({len(obj_ids)}) ...>")
    if obj_ids is None:  # if there are specific obj_ids we use all from a split
        obj_ids = metainfo.obj_ids
        log.info(f"==> selecting obj_ids ({cfg.split}) ...>")

    if cfg.split != "train" and cfg.prior_idx == "train":
        log.info("WARNING! Only select prior_idx for split=train ...>")
        log.info("==> Set prior_idx=False ...>")
        cfg.prior_idx = "mean"

    if not cfg.eval and cfg.save_mesh:
        log.info("WARNING! The mesh is only created in the eval loop ...>")
        log.info("==> Set eval=True ...>")
        cfg.eval = True

    latents = []
    for obj_id in tqdm(obj_ids):
        log.info(f"==> initializing datamodule <{cfg.data._target_}>")
        cfg.data.obj_id = obj_id
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

        log.info(f"==> initializing model <{cfg.model._target_}>")
        cfg.model.prior_obj_id = obj_id
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        log.info("==> initializing callbacks ...")
        callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

        log.info("==> initializing logger ...")
        logger: LightningLogger = hydra.utils.instantiate(cfg.logger)

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
            log.info(f"==> optimize {obj_id=} ...")
            trainer.fit(model=model, datamodule=datamodule)

        # finish the wandb run in order to track all the optimizations seperate
        latents.append(model.latent)
        wandb.finish()

        if cfg.create_video and isinstance(logger, WandbLogger):
            log.info("==> creating video ...")
            path = Path(cfg.paths.video_dir)
            path.mkdir(parents=True, exist_ok=True)
            create_video(video_dir=path, obj_id=obj_id)

    # create the meshes
    meshes = []
    if cfg.eval or cfg.save_mesh:
        log.info("==> create meshes ...")
        for obj_id in obj_ids:
            mesh = model.to_mesh()
            meshes.append(mesh)
            if cfg.save_mesh:
                path = Path(cfg.paths.mesh_dir, f"{obj_id}.obj")
                path.parent.mkdir(parents=True, exist_ok=True)
                o3d.io.write_triangle_mesh(
                    path.as_posix(),
                    mesh=mesh,
                    write_triangle_uvs=False,
                )

    # evaluate the generated 3D shapes
    if cfg.eval:
        log.info("==> start evalution ...")
        model.deepsdf.eval()
        chamfer_distance = ChamferDistance(num_samples=2048)
        earth_movers_distance = EarthMoversDistance(num_samples=2048)
        fechet_inception_distance = FrechetInceptionDistance(feature=2048)

        # get the metric statistics for all of the objects
        for latent, obj_id, mesh in tqdm(zip(latents, obj_ids, meshes)):
            model.latent = latent
            surface_samples = metainfo.load_surface_samples(obj_id=obj_id)
            chamfer_distance.update(mesh, surface_samples)
            earth_movers_distance.update(mesh, surface_samples)

        log.info("==> save metrics ...")
        metrics = {
            "chamfer_distance": chamfer_distance.compute(),
            "earth_movers_distance": earth_movers_distance.compute(),
        }
        df = pd.DataFrame([metrics])
        df.to_csv(cfg.paths.metrics_path, index=False)
