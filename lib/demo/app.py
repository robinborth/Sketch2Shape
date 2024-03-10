import json
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from lib.data.dataset.optimize_latent import InferenceOptimizerDataset
from lib.demo.utils import (
    create_model,
    real_time_inference,
    st_canvas_to_silhouette,
    st_canvas_to_sketch,
)

#################################################################
# Static configs
#################################################################

checkpoints_folder = "/home/borth/sketch2shape/checkpoints/"
temp_folder = "/home/borth/sketch2shape/temp/"

#################################################################
# States
#################################################################

if "silhouettes" not in st.session_state:
    st.session_state.silhouettes = []

if "optimize_images" not in st.session_state:
    st.session_state.optimize_image = None

if "azims" not in st.session_state:
    st.session_state.azims = []

if "elevs" not in st.session_state:
    st.session_state.elevs = []


def add_silhouette(silhouette):
    st.session_state.silhouettes.append(silhouette)
    st.session_state.azims.append(st.session_state.azim)
    st.session_state.elevs.append(st.session_state.elev)


def clear_silhouettes():
    st.session_state.silhouettes = []
    st.session_state.azims = []
    st.session_state.elevs = []


#################################################################
# Settings
#################################################################


with st.sidebar:
    st.title("Sketch2Shape")
    with st.expander("Settings", True):
        stroke_width = st.slider("Stroke width: ", 1, 25, 3)

        _modes = {"\u270E Pencil": "black", "\u2716 Eraser": "white"}
        stroke_mode = st.selectbox("Stroke mode:", _modes.keys())
        stroke_color = "black" if stroke_mode not in _modes else _modes[stroke_mode]

        azim = st.slider("Azim: ", -180, 180, 40, key="azim")
        elev = st.slider("Elev: ", -90, 90, -20, key="elev")
        zoom = st.slider("Zoom: ", 0.5, 2.0, 1.0)

        realtime_update = st.checkbox("Update in realtime", True)

    input_output_expander = st.expander("Input/Output", False)
    with input_output_expander:
        # checkpoints
        deepsdf_ckpt = st.selectbox("DeepSDF checkpoint", ["deepsdf.ckpt"])
        deepsdf_ckpt_path = Path(checkpoints_folder, deepsdf_ckpt or "deepsdf.ckpt")
        loss_ckpt = st.selectbox(
            "Encoder checkpoint",
            ["latent_synthetic.ckpt", "latent_rendered.ckpt", "latent_traverse.ckpt"],
            index=2,
        )
        loss_ckpt_path = Path(checkpoints_folder, loss_ckpt or "latent_traverse.ckpt")
        # Settings for the background
        bg_image = st.file_uploader("Upload background:", type=["png", "jpg"])
        default_background = v2.functional.to_pil_image(np.ones((256, 256, 3)))
        background_image = Image.open(bg_image) if bg_image else default_background
        # Settings for the sketch
        sketch_json_data = st.file_uploader("Upload sketch:", type=["json"])
        initial_drawing = json.load(sketch_json_data) if sketch_json_data else None
        file_name = st.text_input("Filename:", value="sketch_0")

    with st.expander("Debug", False):
        inference_model = st.checkbox("Inference model", True)
        visualize_model_input = st.checkbox("Visualize model input/output", False)

#################################################################
# Inference Model
#################################################################
model = create_model(
    loss_ckpt_path=str(loss_ckpt_path),
    deepsdf_ckpt_path=str(deepsdf_ckpt_path),
)
model.deepsdf.create_camera(azim=azim, elev=elev, focal=int(zoom * 512))

#################################################################
# Drawable Canvas
#################################################################

real_time_col1, real_time_col2 = st.columns(2)
debug_col1, debug_col2, debug_col3, debug_col4, debug_col5 = st.columns(5)
with real_time_col1:
    st.text("Sketch:")
    main_canvas = st_canvas(
        initial_drawing=initial_drawing,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=background_image,
        update_streamlit=realtime_update,
        width=256,
        height=256,
        drawing_mode="freedraw",
        key="main_canvas",
    )
main_sketch = st_canvas_to_sketch(main_canvas)
valid_main_sketch = main_sketch is not None and main_sketch.min() < 255
with debug_col1:
    if valid_main_sketch and visualize_model_input:
        st.text("Sketch Output:")
        st.image(main_sketch)
with input_output_expander:
    download_sketch_button = st.download_button(
        label="Download Sketch",
        file_name=f"{str(file_name)}.json",
        mime="application/json",
        data=json.dumps(main_canvas.json_data),
    )

    if st.button("Prepare mesh"):
        with st.spinner("Preparing mesh..."):
            mesh = model.to_mesh()
        path = Path(temp_folder, "tmp_mesh.obj")
        o3d.io.write_triangle_mesh(str(path), mesh=mesh, write_triangle_uvs=False)
        with open(path) as data:
            download_sketch_button = st.download_button(
                label="Download Mesh",
                data=data,
                file_name=f"{str(file_name)}.obj",
            )


#################################################################
# Real-Time Inference and Silhouette Canvas
#################################################################

silhouette = None
if valid_main_sketch and inference_model:
    sketch_input = model.sketch_transform(main_sketch)[None, ...]
    out = real_time_inference(model, sketch_input)
    background = (out["normals"] * 255).astype(np.uint8)
    with real_time_col2:
        st.text("Rendered Normal:")
        silhouette_canvas = st_canvas(
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=Image.fromarray(background),
            update_streamlit=realtime_update,
            width=256,
            height=256,
            drawing_mode="freedraw",
            key="silhouette_canvas",
        )
        silhouette = st_canvas_to_silhouette(silhouette_canvas, background=background)

    if visualize_model_input:
        with debug_col2:
            st.text("Model Input:")
            st.image(out["model_input"])
        with debug_col3:
            st.text("Rendered Grayscale:")
            st.image(out["grayscale"])
        with debug_col4:
            st.text("Rendered SDF:")
            st.image(out["sdf"])
        with debug_col5:
            st.text("Silhouette:")
            st.image(out["silhouette"])


#################################################################
# Silhouette Options
#################################################################

if valid_main_sketch:
    if st.button("Save Silhouette"):
        add_silhouette(silhouette)
    if st.button("Clear Silhouette"):
        clear_silhouettes()


#################################################################
# Silhouettes
#################################################################

if st.session_state.silhouettes:
    st.text("Silhouettes:")
    n_cols = len(st.session_state.silhouettes)
    for sil, col in zip(st.session_state.silhouettes, st.columns(n_cols)):
        with col:
            st.image(sil)

    if st.button("Optimize"):
        silhouettes = [Image.fromarray(sil) for sil in st.session_state.silhouettes]
        dataset = InferenceOptimizerDataset(
            sketch=Image.fromarray(main_sketch),
            silhouettes=silhouettes,
            azims=st.session_state.azims,
            elevs=st.session_state.elevs,
        )
        dataloader = DataLoader(dataset, batch_size=1)
        optimizer = model.configure_optimizers()
        max_epochs = 10
        for epoch in range(max_epochs):
            for idx, batch in enumerate(dataloader):
                for key in batch.keys():
                    batch[key] = batch[key].to(model.device)
                loss = model.training_step(batch, idx)
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            st.text("Optimize Results:")
            optimize_image = model.capture_camera_frame()
            st.image(optimize_image)
