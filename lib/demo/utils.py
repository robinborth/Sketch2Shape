import hydra
import numpy as np
import streamlit as st
import torch
from torchvision.transforms import v2

from lib.optimizer.sketch import SketchOptimizer
from lib.utils.config import load_config


@st.cache_resource()
def create_model(loss_ckpt_path: str, deepsdf_ckpt_path: str) -> SketchOptimizer:
    cfg = load_config("optimize_sketch", ["+dataset=shapenet_chair_4096"])
    cfg.model.latent_init = "mean"
    cfg.loss_ckpt_path = loss_ckpt_path
    cfg.deepsdf_ckpt_path = deepsdf_ckpt_path
    model = hydra.utils.instantiate(cfg.model).to("cuda")
    return model


def real_time_inference(model, sketch):
    model.latent = model.loss.embedding(sketch, mode="sketch")[0]
    model_input = model.deepsdf.loss_input_to_image(sketch)
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
    return {
        "normals": normals.detach().cpu().numpy(),
        "grayscale": grayscale.detach().cpu().numpy(),
        "silhouette": silhouette["final_silhouette"].detach().cpu().numpy(),
        "model_input": model_input.detach().cpu().numpy(),
    }


def st_canvas_to_sketch(canvas_result):
    padding = 0.1
    if canvas_result.image_data is not None:
        if (canvas_result.image_data.sum()) == 0:
            return None
        channel = torch.tensor(255 - canvas_result.image_data[:, :, 3])
        sketch = torch.stack([channel] * 3, dim=0)  # (3, H, W)

        # center the sketch
        mask = sketch.sum(0) < 255
        idx = torch.where(mask)
        bbox = sketch[:, idx[0].min() : idx[0].max(), idx[1].min() : idx[1].max()]

        # add padding
        max_size = max(bbox.shape[1], bbox.shape[2])
        pad_2 = (max_size - bbox.shape[2]) // 2
        pad_1 = (max_size - bbox.shape[1]) // 2
        bbox = torch.nn.functional.pad(bbox, (pad_2, pad_2, pad_1, pad_1), value=255)
        margin = int(max_size * padding)
        bbox = torch.nn.functional.pad(
            bbox, (margin, margin, margin, margin), value=255
        )
        sketch = v2.functional.resize(bbox, (256, 256))
        sketch = sketch.permute(1, 2, 0)  # (H, W, 3)
        return sketch.detach().cpu().numpy()

    return None
