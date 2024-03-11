from functools import partial
from typing import Any

import lightning as L
import numpy as np
import streamlit as st
import torch
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from lib.data.dataset.optimize_latent import InferenceOptimizerDataset
from lib.optimizer.sketch import SketchOptimizer


@st.cache_resource()
def create_model(
    loss_ckpt_path: str,
    deepsdf_ckpt_path: str,
) -> SketchOptimizer:
    model = SketchOptimizer(
        loss_ckpt_path=loss_ckpt_path,
        deepsdf_ckpt_path=deepsdf_ckpt_path,
        latent_init="mean",
        loss_mode="none",
        reg_loss="latent",
        reg_weight=1e-02,
        silhouette_loss="silhouette",
        silhouette_weight=1.0,
        optimizer=partial(Adam, lr=1e-02),
        verbose=False,
    ).to("cuda")
    return model


def optimize_model(
    model: SketchOptimizer,
    sketch: Any,
    azims: list,
    elevs: list,
    silhouettes: list[Any],
    temp_folder: str,
):
    print("==> initializing dataset <{cfg.data._target_}>")
    dataset = InferenceOptimizerDataset(
        sketch=sketch,
        silhouettes=silhouettes,
        azims=azims,
        elevs=elevs,
    )
    dataloader = DataLoader(dataset, batch_size=1)

    print("==> initializing logger ...")
    logger = L.pytorch.loggers.wandb.WandbLogger(
        save_dir=temp_folder,
        project="sketch2shape",
        entity="research-korth",
        tags=["optimize_sketch", "inference"],
    )

    print("==> initializing trainer ...")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=10,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        min_epochs=1,
        accumulate_grad_batches=1,
        logger=logger,
    )

    print("==> start optimization ...")
    trainer.fit(model, dataloader)
    wandb.finish()


def real_time_inference(model, sketch=None):
    out = {}
    if sketch is not None:
        # model = model.to("cuda")
        model.latent = model.loss.embedding(sketch.to("cuda"), mode="sketch")[0]
        model_input = model.deepsdf.loss_input_to_image(sketch)
        out["model_input"] = model_input.detach().cpu().numpy()
    with torch.no_grad():
        points, surface_mask = model.deepsdf.sphere_tracing(
            latent=model.latent,
            points=model.deepsdf.camera_points,
            mask=model.deepsdf.camera_mask,
            rays=model.deepsdf.camera_rays,
        )
        normals = model.deepsdf.render_normals(
            points=points,
            latent=model.latent,
            mask=surface_mask,
        )
        grayscale = model.deepsdf.normal_to_grayscale(normal=normals)
        silhouette = model.deepsdf.render_silhouette(
            normals=normals,
            points=points,
            latent=model.latent,
            proj_blur_eps=0.7,
            weight_blur_kernal_size=9,
            weight_blur_sigma=9.0,
        )
    out.update(
        {
            "normal": normals.detach().cpu().numpy(),
            "grayscale": grayscale.detach().cpu().numpy(),
            "sdf": silhouette["min_sdf"].detach().cpu().numpy(),
            "silhouette": silhouette["final_silhouette"].detach().cpu().numpy(),
        }
    )
    return out


def center_with_padding(
    image: np.ndarray,
    padding: float = 0.1,
    threshold: int = 600,
) -> np.ndarray:
    """
    Centers an image of dim (H, W, 3) with a padding and values uint8 (0, 255). Where
    white is #FFFFFF and black #000000, hence (255,255,255) and (0,0,0).
    """
    # extract the backgound and sketch info
    sketch = torch.tensor(image, dtype=torch.uint8)
    background_mask = sketch.sum(-1) > threshold
    if not (~background_mask).sum().item():
        return np.full((256, 256, 3), 255, dtype=np.uint8)

    # center
    idx = torch.where(~background_mask)  # actual drawing
    bbox = sketch[idx[0].min() : idx[0].max(), idx[1].min() : idx[1].max(), :]
    max_size = max(bbox.shape[0], bbox.shape[1])
    pad_1 = (max_size - bbox.shape[1]) // 2
    pad_0 = (max_size - bbox.shape[0]) // 2
    bbox = torch.nn.functional.pad(bbox, (0, 0, pad_1, pad_1, pad_0, pad_0), value=255)

    # add padding
    m = int(max_size * padding)
    bbox = torch.nn.functional.pad(bbox, (0, 0, m, m, m, m), value=255)

    # resize
    sketch = v2.functional.resize(bbox.permute(2, 0, 1), (256, 256)).permute(1, 2, 0)
    return sketch.detach().cpu().numpy()


def st_canvas_to_sketch(canvas_result):
    if (image_data := canvas_result.image_data) is None:
        return None
    strokes = image_data[:, :, 3] > 0  # (H, W) with no color values (0, 255)
    white_strokes = image_data[:, :, :3].sum(-1) > 600  # (H, W)
    sketch_mask = strokes & (~white_strokes)  # (H, W)
    sketch = 255 - image_data[:, :, 3]  # convert to color values black=0
    sketch[~sketch_mask] = 255  # set everything that is no sketch to white
    sketch = np.stack([sketch] * 3, axis=-1)  # (H, W, 3)
    return center_with_padding(sketch)


def st_canvas_to_silhouette(canvas_result, background=None):
    if (image_data := canvas_result.image_data) is None:
        return None
    strokes = image_data[:, :, 3] > 0  # (H, W) with no color values (0, 255)
    white_strokes = image_data[:, :, :3].sum(-1) > 600  # (H, W)
    sketch_mask = strokes & (~white_strokes)  # (H, W)
    background[sketch_mask] = (255 - image_data[:, :, 3])[sketch_mask, None]
    background[white_strokes] = 255  # set everything that is no sketch to white
    return background
