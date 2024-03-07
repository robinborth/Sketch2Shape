import json

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from torchvision.transforms import v2

from lib.data.transforms import SketchTransform
from lib.demo.utils import create_model, real_time_inference, st_canvas_to_sketch

with st.sidebar:
    st.title("Sketch2Shape")
    input_output_expander = st.expander("Input/Output", True)
    with input_output_expander:
        # Settings for the background
        bg_image = st.file_uploader("Upload background:", type=["png", "jpg"])
        default_background = v2.functional.to_pil_image(np.ones((256, 256, 3)))
        background_image = Image.open(bg_image) if bg_image else default_background
        # Settings for the sketch
        sketch_json_data = st.file_uploader("Upload sketch:", type=["json"])
        initial_drawing = json.load(sketch_json_data) if sketch_json_data else None

    with st.expander("Settings", True):
        stroke_width = st.slider("Stroke width: ", 1, 25, 3)
        stroke_color = st.color_picker("Stroke color hex: ")
        azim = st.slider("Azim: ", -180, 180, 40)
        elev = st.slider("Elev: ", -90, 90, -20)
        zoom = st.slider("Zoom: ", 0.5, 2.0, 1.0)
        realtime_update = st.checkbox("Update in realtime", True)

# Create a canvas component
st.text("Sketch:")
canvas_result = st_canvas(
    initial_drawing=initial_drawing,
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_image=background_image,
    update_streamlit=realtime_update,
    width=256,
    height=256,
    drawing_mode="freedraw",
    key="canvas",
)
sketch = st_canvas_to_sketch(canvas_result)
with input_output_expander:
    file_name = st.text_input("Filename:", value="sketch_0")
    button = st.download_button(
        label="Download Sketch",
        file_name=f"{str(file_name)}.json",
        mime="application/json",
        data=json.dumps(canvas_result.json_data),
    )


# Settup the Model
transform = SketchTransform()
model = create_model(
    loss_ckpt_path="/home/borth/sketch2shape/checkpoints/latent_traverse.ckpt",
    deepsdf_ckpt_path="/home/borth/sketch2shape/checkpoints/deepsdf.ckpt",
)
model.deepsdf.create_camera(azim=azim, elev=elev, focal=int(zoom * 512))


# Update the Model Output
if sketch is not None:
    sketch_input = transform(sketch)[None, ...].to("cuda")
    out = real_time_inference(model, sketch_input)
    # draw the input
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text("Model Input:")
        st.image(out["model_input"])
    with col2:
        st.text("Rendered Normal:")
        st.image(out["normals"])
    with col3:
        st.text("Rendered Grayscale:")
        st.image(out["grayscale"])
