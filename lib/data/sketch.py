import os
from typing import Union

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes


def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        # only render RGB channels
        ax.imshow(im)
        if not show_axes:
            ax.set_axis_off()


def render_shapenet(
    path: str,
    dist: float = 1.0,
    elev: Union[int, torch.Tensor] = 0,
    azim: Union[int, torch.Tensor] = 0,
    color: float = 0.85,
    image_size: int = 256,
    device: str = "cuda",
):
    # load and prepare the shapenet objects
    verts, faces_idx, _ = load_obj(path, load_textures=False)
    faces = faces_idx.verts_idx

    # set the rendering texture color to grey
    verts_rgb = torch.zeros_like(verts)[None]  # (1, V, 3)
    verts_rgb[:, :, :] = color
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # create a Meshes object
    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures,
    )

    # get the correct batch_size to cast the mash
    if torch.is_tensor(elev) and torch.is_tensor(azim):
        assert elev.shape[0] == azim.shape[0]  # type: ignore

    batch_size = 1
    if torch.is_tensor(elev):
        batch_size = elev.shape[0]  # type: ignore
    elif torch.is_tensor(azim):
        batch_size = azim.shape[0]  # type: ignore

    mesh = mesh.extend(batch_size)

    # initilizing the camara
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # initilizing the rasterization
    raster_settings = RasterizationSettings(image_size=image_size, bin_size=0)

    # initilizing the lights
    lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])

    # initilizing the renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
        ),
    )

    # create the rendered image without the depth
    images = renderer(mesh).cpu().numpy()[..., :3]
    return (images * 255).astype(np.uint8)


def image_to_sketch(
    images,
    t_lower: int = 100,
    t_upper: int = 150,
    aperture_size: int = 3,  # 3, 5, 7
    L2gradient: bool = True,
):
    edges = []
    for img in images:
        edge = cv.Canny(img, t_lower, t_upper, L2gradient=L2gradient)
        edge = cv.bitwise_not(edge)
        edges.append(edge)
    edges = np.stack((np.stack(edges),) * 3, axis=-1)
    return edges


def default_elev_azim():
    elev = torch.tensor([-22.5, 0, 22.5, 45.0, 90])
    elev = elev.view(5, 1).repeat(1, 10).view(-1)

    azim = torch.tensor([-140, -100, -60, -20, 0, 20, 60, 100, 140, 180])
    azim = azim.repeat(5)

    return elev, azim


def cartesian_elev_azim(elev, azim):
    _elev = torch.tensor(elev).view(len(elev), 1).repeat(1, len(azim)).view(-1)
    _azim = torch.tensor(azim).repeat(len(elev))
    return _elev, _azim


def obj_path(obj_id: str, config) -> str:
    return os.path.join(config.dataset_path, obj_id, "model_normalized.obj")


def sketches_folder(obj_id: str, config):
    return os.path.join(config.dataset_path, obj_id, "sketches")


def images_folder(obj_id: str, config):
    return os.path.join(config.dataset_path, obj_id, "images")


def sketch_path(obj_id: str, index: int, config):
    return os.path.join(config.dataset_path, obj_id, f"sketches/{index:05}.jpg")


def image_path(obj_id: str, index: int, config):
    return os.path.join(config.dataset_path, obj_id, f"images/{index:05}.jpg")
