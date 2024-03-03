import hydra
import numpy as np
import streamlit as st

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


def st_canvas_to_sketch(canvas_result):
    if canvas_result.image_data is not None:
        channel = 255 - canvas_result.image_data[:, :, 3]
        return np.stack([channel] * 3, axis=-1)
    return None
