from logging import Logger
from pathlib import Path

import hydra
import lightning as L
import open3d as o3d
import pandas as pd
import torch
import wandb
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger as LightningLogger
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from lib.data.metainfo import MetaInfo
from lib.data.transforms import SketchTransform
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
    elif obj_dir := cfg.get("obj_dir"):
        obj_ids = [str(path.resolve()) for path in Path(obj_dir).iterdir()]
        log.info(f"==> selecting specified sketches ({len(obj_ids)}) ...>")
    else:
        obj_ids = metainfo.obj_ids
        log.info(f"==> selecting obj_ids ({cfg.split}) ...>")

    latents = []
    retrieval_idxs = []
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
        if cfg.model.latent_init == "retrieval":
            retrieval_idxs.append(model.retrieval_idx[0])

    # create the meshes
    meshes = []
    if cfg.eval or cfg.save_mesh:
        log.info("==> create meshes ...")
        for obj_id in obj_ids:
            if cfg.model.latent_init == "retrieval":
                mesh = metainfo.load_normalized_mesh(obj_id=obj_id)
            else:
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

    # save the optimized latents
    if cfg.save_latent:
        for frame_idx, latent in enumerate(latents):
            path = Path(cfg.paths.latent_dir, f"{frame_idx:05}.pt")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(latent.detach().cpu(), path)

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
        sketch_transform = SketchTransform(normalize=False)
        model.deepsdf.create_camera(azim=eval_azim, elev=eval_elev)
        log.info("==> start evaluate FID and CLIPScore ...")
        for idx, obj_id in tqdm(enumerate(obj_ids), total=len(obj_ids)):
            # gt sketch
            label = metainfo.obj_id_to_label(obj_id)
            gt_sketch = metainfo.load_image(label, eval_view_id, 0)
            gt_sketch = sketch_transform(gt_sketch)[None, ...]
            if cfg.model.latent_init == "retrieval":
                label = retrieval_idxs[idx]
                rendered_sketch = metainfo.load_image(label, eval_view_id, 2)
                rendered_sketch = sketch_transform(rendered_sketch)[None, ...]
            else:
                # rendered sketch
                model.latent = latents[idx]
                rendered_normal = model.capture_camera_frame("grayscale")
                rendered_normal = rendered_normal.permute(2, 0, 1).detach().cpu()
                rendered_sketch = sketch_transform(rendered_normal)[None, ...]
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
