import streamlit as st
import torch
from streamlit_drawable_canvas import st_canvas

from lib.data.transforms import SketchTransform
from lib.demo.utils import create_model, st_canvas_to_sketch

# Settings For Visulisation
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
azim = st.sidebar.slider("Azim: ", -180, 180, 40)
elev = st.sidebar.slider("Elev: ", -90, 90, -20)
realtime_update = st.sidebar.checkbox("Update in realtime", True)


model = create_model(
    loss_ckpt_path="/home/borth/sketch2shape/checkpoints/latent_traverse.ckpt",
    deepsdf_ckpt_path="/home/borth/sketch2shape/checkpoints/deepsdf.ckpt",
)
transform = SketchTransform()
model.deepsdf.create_camera(azim=azim, elev=elev)

# Create a canvas component
col1, col2, col3 = st.columns(3)
with col1:
    canvas_result = st_canvas(
        stroke_width=stroke_width,
        update_streamlit=realtime_update,
        width=256,
        height=256,
        drawing_mode="freedraw",
        key="canvas",
    )
    button = st.button("clear_canvas")
    if button:
        print("clear")
sketch = st_canvas_to_sketch(canvas_result)

# Update the model
if sketch is not None:
    sketch_input = transform(sketch)[None, ...].to("cuda")
    model.latent = model.loss.embedding(sketch_input, mode="sketch")[0]

    # draw the input
    # st.image(sketch)
    with col2:
        st.image(model.deepsdf.loss_input_to_image(sketch_input).detach().cpu().numpy())

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
        # grayscale = model.deepsdf.normal_to_grayscale(normal=normals)
        silhouette = model.deepsdf.render_silhouette(
            normals=normals,
            points=points,
            latent=model.latent,
            proj_blur_eps=0.7,
            weight_blur_kernal_size=9,
            weight_blur_sigma=9.0,
        )

    with col3:
        st.image(normals.detach().cpu().numpy())

    # with col1:
    #     st.image(normals.detach().cpu().numpy())
    # with col2:
    #     final_silhouette = 1 - silhouette["final_silhouette"]
    #     st.image(final_silhouette.detach().cpu().numpy())
    # with col3:
    #     min_sdf = 1 - silhouette["min_sdf"]
    #     st.image(min_sdf.detach().cpu().numpy())
