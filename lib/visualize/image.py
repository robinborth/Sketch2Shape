import matplotlib.pyplot as plt
import numpy as np


def visualize_mask(camera, mask):
    mask_image = mask.view(camera.resolution, camera.resolution).detach().cpu().numpy()
    plt.imshow(mask_image.T)


def visualize_image(image):
    image = image.detach().cpu().numpy()
    plt.imshow(image)
    plt.show()
    return image


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
    plt.clf()
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
    return fig


def transform_to_plot(data, batch=False):
    if batch:
        data = np.transpose(data, (0, 2, 3, 1))
    else:
        data = np.transpose(data, (1, 2, 0))
    return np.clip(data, 0, 1)


def plot_single_image(image):
    plt.clf()
    plt.figure(figsize=(2, 2))
    plt.imshow(image)
    plt.axis("off")
    plt.show()
