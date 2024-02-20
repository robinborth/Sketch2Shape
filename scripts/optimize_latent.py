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
from lib.data.transforms import BaseTransform, ToSketch
from lib.eval.chamfer_distance import ChamferDistance
from lib.eval.clip_score import CLIPScore
from lib.eval.earth_movers_distance import EarthMoversDistance
from lib.render.utils import create_video
from lib.utils.config import instantiate_callbacks, log_hyperparameters


def optimize_latent(cfg: DictConfig, log: Logger) -> None:
    log.info("==> loading config ...")
    # make sure that the batch_size is 1
    assert cfg.data.batch_size == 1
    L.seed_everything(cfg.seed)
    metainfo = MetaInfo(data_dir=cfg.data.data_dir, split=cfg.split)

    # specific obj_ids are selected
    if obj_ids := cfg.get("obj_ids"):
        log.info(f"==> selecting specified obj_ids ({len(obj_ids)}) ...>")
    # obj_ids from the selected split
    if obj_ids is None:
        obj_ids = metainfo.obj_ids
        log.info(f"==> selecting obj_ids ({cfg.split}) ...>")

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

        # finish the wandb run to track all the optimizations seperatly
        wandb.finish()

        if cfg.create_video and isinstance(logger, WandbLogger):
            log.info("==> creating video ...")
            path = Path(cfg.paths.video_dir)
            path.mkdir(parents=True, exist_ok=True)
            create_video(video_dir=path, obj_id=obj_id)

        # update the latents from the current obj_id
        latents.append(model.latent)

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
    eval_view_id: int = 11
    eval_azim: float = 40
    eval_elev: float = -30
    if cfg.eval:
        model.deepsdf.eval()
        cd = ChamferDistance(num_samples=2048)
        emd = EarthMoversDistance(num_samples=2048)
        fid = FrechetInceptionDistance(feature=2048, normalize=True)
        clip = CLIPScore()

        log.info("==> start evaluate CD and EMD ...")
        for obj_id, mesh in tqdm(zip(obj_ids, meshes), total=len(obj_ids)):
            surface_samples = metainfo.load_surface_samples(obj_id=obj_id)
            cd.update(mesh, surface_samples)
            emd.update(mesh, surface_samples)

        # frechet inception distance and clip score
        transform = BaseTransform(normalize=False)
        to_sketch = ToSketch()
        model.deepsdf.create_camera(azim=eval_azim, elev=eval_elev)
        log.info("==> start evaluate FID and CLIPScore ...")
        for latent, obj_id in tqdm(zip(latents, obj_ids), total=len(obj_ids)):
            # gt sketch
            gt_sketch = metainfo.load_sketch(obj_id, f"{eval_view_id:05}")
            gt_sketch = transform(gt_sketch)[None, ...]
            # rendered sketch
            model.latent = latent
            rendered_normal = model.capture_camera_frame().permute(2, 0, 1)
            rendered_sketch = to_sketch(rendered_normal.detach().cpu())[None, ...]
            # frechet inception distance
            fid.update(gt_sketch, real=True)
            fid.update(rendered_sketch, real=False)
            # clip score
            clip.update(gt_sketch, rendered_sketch)

        log.info("==> save metrics ...")
        metrics = {
            "CD": cd.compute().item(),
            "EMD": emd.compute().item(),
            "FID": fid.compute().item(),
            "CLIPScore": clip.compute().item(),
        }
        df = pd.DataFrame([metrics])
        df.to_csv(cfg.paths.metrics_path, index=False)
