from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE


def transform_to_plot(data, batch=False):
    if batch:
        data = np.transpose(data, (0, 2, 3, 1))
    else:
        data = np.transpose(data, (1, 2, 0))
    data = (data * 0.5) + 0.5
    return np.clip(data, 0, 1)


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
    rcol = lambda _: np.random.randint(0, 256)
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
