from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE


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
    return fig


def transform_to_plot(data, batch=False):
    if batch:
        data = np.transpose(data, (0, 2, 3, 1))
    else:
        data = np.transpose(data, (1, 2, 0))
    return np.clip(data, 0, 1)


def plot_single_image(data):
    data = transform_to_plot(data)
    plt.figure(figsize=(2, 2))
    plt.imshow(data)
    plt.axis("off")
    plt.show()


def show_sketch(dataset, idx):
    image_data = transform_to_plot(dataset[idx]["sketch"], batch=False)
    plot_single_image(image_data)


def show_sketches(dataset, idx):
    image_data = transform_to_plot(dataset[idx]["sketch"], batch=True)
    image_grid(image_data, 4, 8)


def show_rendered_images(dataset, idx):
    image_data = transform_to_plot(dataset[idx]["image"], batch=True)
    image_grid(image_data, 4, 8)


def plot_top_32(
    metainfo,
    dataset,
    batch,
    image_id,
):
    idx = np.where(batch["image_ids"] == image_id)[0][0]
    gt_label = batch["labels"][idx]
    # print(f"{gt_label=}")
    gt_image_id = batch["image_ids"][idx]
    # print(f"{gt_image_id=}")
    labels_at_1_object = batch["labels_at_1_object"][idx]
    # print(f"{labels_at_1_object=}")
    image_ids_at_1_object = batch["image_ids_at_1_object"][idx]
    # print(f"{image_ids_at_1_object=}")

    obj_id = metainfo.label_to_obj_id(gt_label)
    image_id = str(gt_image_id).zfill(5)
    sketch = dataset._fetch("sketches", obj_id, image_id)
    plot_single_image(sketch)

    images = []
    for label, image_id in zip(labels_at_1_object, image_ids_at_1_object):
        is_false = gt_label != label
        obj_id = metainfo.label_to_obj_id(label)
        image_id = str(image_id).zfill(5)
        image = dataset._fetch("images", obj_id, image_id)
        if is_false:
            image[image > 0.95] = 0
        images.append(image)
    image_data = transform_to_plot(images, batch=True)
    image_grid(image_data, 4, 8)


def get_all_vectors_from_faiss_index(index):
    num_vectors = index.ntotal
    all_vectors = np.empty((num_vectors, index.d), dtype=np.float32)
    batch_size = 1000
    for start in range(0, num_vectors, batch_size):
        end = min(start + batch_size, num_vectors)
        vectors_batch = index.reconstruct_n(start, end - start)
        all_vectors[start:end, :] = vectors_batch
    return all_vectors


def batch_outputs(data):
    out = defaultdict(list)
    for outputs in data:
        for key, value in outputs.items():
            out[key].append(value)
    for key, value in out.items():
        out[key] = np.concatenate(value)  # type: ignore
    return out


def detach_batch_output(batch, output):
    return {
        "sketch_emb": output["sketch_emb"].detach().cpu().numpy(),
        "image_emb": output["image_emb"].detach().cpu().numpy(),
        "label": batch["label"].detach().cpu().numpy(),
        "sketch": batch["sketch"].detach().cpu().numpy(),
        "image": batch["image"].detach().cpu().numpy(),
    }


def tsne(X, labels):
    X_embedded = TSNE(
        n_components=2,
        learning_rate="auto",
        init="random",
        perplexity=3,
    ).fit_transform(X)

    # Create a DataFrame with t-SNE results and class labels
    classes = [f"Class {c}" for c in labels]
    data = {"X0": X_embedded[:, 0], "X1": X_embedded[:, 1], "Class": classes}
    df = pd.DataFrame(data)

    # Define a color map for each class
    rcol = lambda x: np.random.randint(0, 256)
    color_map = {
        class_label: f"rgb({rcol()}, {rcol()}, {rcol()})"
        for class_label in df["Class"].unique()
    }

    # Set the size of the Plotly figure and use the color map
    fig = px.scatter(df, x="X0", y="X1", color="Class", title="t-SNE Visualization")
    #  labels={'Class': 'Class'}, color_discrete_sequence=color_map)

    # Set the width and height of the figure (you can adjust these values)
    fig.update_layout(width=600, height=600)

    return fig
