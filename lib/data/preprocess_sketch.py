import os
from typing import Any

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from open3d.utility import VerbosityLevel, set_verbosity_level
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    DirectionalLights,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    HardFlatShader,
    HardGouraudShader,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftGouraudShader,
    SoftPhongShader,
    SoftSilhouetteShader,
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
    elevs: list[int] = [0],
    azims: list[int] = [0],
    color: float = 1,
    image_size: int = 256,
    device: Any = None,
):
    # settings
    set_verbosity_level(VerbosityLevel.Error)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the mesh
    open3d_mesh = o3d.io.read_triangle_mesh(path)
    verts = torch.tensor(np.asarray(open3d_mesh.vertices), dtype=torch.float32).to(
        device
    )
    faces = torch.tensor(np.asarray(open3d_mesh.triangles), dtype=torch.int64).to(
        device
    )

    # set the texture
    verts_rgb = torch.ones_like(verts)[None]
    verts_rgb[:, :, :] = color
    textures = TexturesVertex(verts_features=verts_rgb)

    # create pytorch 3d mesh
    mesh = Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures,
    )

    # initilizing the rasterization
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=8,
    )

    # initilizing the lights
    lights = PointLights(device=device, location=[[-1, 1, -2]])

    # initilizing the renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            raster_settings=raster_settings,
        ),
        shader=SoftPhongShader(
            device=device,
            lights=lights,
        ),
    )

    # initilize the transformation delta for the object
    delta = ((verts.max(dim=0)[0] + verts.min(dim=0)[0]) / 2).detach().cpu()

    # loop for each camara position
    images = []
    for elev, azim in zip(elevs, azims):
        R, T = look_at_view_transform(dist=1, elev=elev, azim=azim)
        T[0] -= delta
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        image = renderer(mesh, cameras=cameras).cpu().numpy()[0, ..., :3]
        images.append((image * 255).astype(np.uint8))

    return np.stack(images, axis=0)


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
    return os.path.join(config.data.data_dir, obj_id, "model_normalized.obj")


def sketches_folder(obj_id: str, config):
    return os.path.join(config.data.data_dir, obj_id, "sketches")


def images_folder(obj_id: str, config):
    return os.path.join(config.data.data_dir, obj_id, "images")


def sketch_path(obj_id: str, index: int, config):
    return os.path.join(config.data.data_dir, obj_id, f"sketches/{index:05}.jpg")


def image_path(obj_id: str, index: int, config):
    return os.path.join(config.data.data_dir, obj_id, f"images/{index:05}.jpg")
