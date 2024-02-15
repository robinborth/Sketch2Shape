from pathlib import Path

import hydra
import lightning as L
import numpy as np
import open3d as o3d
import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from lib.render.utils import create_video
from lib.utils.config import log_hyperparameters
from lib.utils.logger import create_logger

log = create_logger("traverse_latent")


@hydra.main(version_base=None, config_path="../conf", config_name="traverse_latent")
def optimize(cfg: DictConfig) -> None:
    log.info("==> loading config ...")
    L.seed_everything(cfg.seed)

    log.info(f"==> initializing model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("==> initializing logger ...")
    logger: Logger = hydra.utils.instantiate(cfg.logger)

    log.info(f"==> initializing trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }
    if logger:
        log.info("==> logging hyperparameters ...")
        log_hyperparameters(object_dict)

    log.info("==> start traversal ...")
    steps = np.linspace(0, 1, num=cfg.traversal_steps)
    dataloader = DataLoader(steps, batch_size=1)  # type: ignore
    trainer.validate(model=model, dataloaders=dataloader)

    if cfg.create_mesh and (meshes := model.meshes):
        log.info("==> save meshes ...")
        for step, mesh in enumerate(meshes):
            path = Path(cfg.paths.mesh_dir, f"step={step:03}.obj")
            path.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_triangle_mesh(
                path.as_posix(),
                mesh=mesh,
                write_triangle_uvs=False,
            )

    # finish the wandb run in order to track all the optimizations seperate
    wandb.finish()

    if cfg.create_video and isinstance(logger, WandbLogger):
        log.info("==> creating video ...")
        path = Path(cfg.paths.video_dir)
        path.mkdir(parents=True, exist_ok=True)
        create_video(video_dir=path, obj_id="traverse_latent")


if __name__ == "__main__":
    optimize()
