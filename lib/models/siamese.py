from collections import defaultdict

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from lightning import LightningModule
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.losses import TripletMarginLoss
from sklearn.manifold import TSNE
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights


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
        n_components=2, learning_rate="auto", init="random", perplexity=3
    ).fit_transform(X)

    # Create a DataFrame with t-SNE results and class labels
    classes = [f"Class {c}" for c in labels]
    data = {"X0": X_embedded[:, 0], "X1": X_embedded[:, 1], "Class": classes}
    df = pd.DataFrame(data)

    # Define a color map for each class
    color_map = {
        class_label: f"rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})"
        for class_label in df["Class"].unique()
    }

    # Set the size of the Plotly figure and use the color map
    fig = px.scatter(df, x="X0", y="X1", color="Class", title="t-SNE Visualization")
    #  labels={'Class': 'Class'}, color_discrete_sequence=color_map)

    # Set the width and height of the figure (you can adjust these values)
    fig.update_layout(width=600, height=600)

    return fig


class Siamese(LightningModule):
    def __init__(
        self,
        decoder: torch.nn.Module,
        miner: miners.BaseMiner,
        loss: losses.BaseMetricLossFunction,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # self.decoder = decoder
        self.decoder = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        self.decoder.fc = torch.nn.Linear(in_features=512, out_features=128)
        self.miner = miner
        self.loss = loss

        self.validation_step_outputs = []  # type: ignore
        self.train_step_outputs = []  # type: ignore

    def forward(self, batch):
        sketch_emb = self.decoder(batch["sketch"])
        image_emb = self.decoder(batch["image"])
        return {"sketch_emb": sketch_emb, "image_emb": image_emb}

    def model_step(self, batch):
        output = self.forward(batch)
        miner_output = self.miner(
            embeddings=output["sketch_emb"],
            labels=batch["label"],
            ref_emb=output["image_emb"],
        )
        loss = self.loss(
            embeddings=output["sketch_emb"],
            labels=batch["label"],
            indices_tuple=miner_output,
            ref_emb=output["image_emb"],
        )

        output["miner_count"] = len(miner_output[0])

        # how many triplets
        batch_size = batch["label"].shape[0]
        m = 4
        # m = self.trainer.datamodule.hparams.sampler.keywords["m"]
        max_count = torch.tensor((m - 1) * batch_size * (batch_size - m))
        output["miner_max_count"] = max_count

        # ratio
        output["miner_ratio"] = output["miner_count"] / output["miner_max_count"]

        return output, loss

    def training_step(self, batch, batch_idx):
        output, loss = self.model_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/miner_ratio", output["miner_ratio"], prog_bar=True)
        self.log("train/miner_count", output["miner_count"])
        self.log("train/miner_max_count", output["miner_max_count"]),
        return loss

    def validation_step(self, batch, batch_idx):
        output, loss = self.model_step(batch)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/miner_count", output["miner_count"])
        self.log("val/miner_max_count", output["miner_max_count"]),
        self.log("val/miner_ratio", output["miner_ratio"], prog_bar=True)
        # outputs = detach_batch_output(batch, output)
        # self.validation_step_outputs.append(outputs)
        return loss

    # def on_validation_end(self) -> None:
    #     output = batch_outputs(self.validation_step_outputs)

    #     image = transform_to_plot(output["image"][0])
    #     sketch = transform_to_plot(output["sketch"][0])
    #     self.logger.log_image(key="val/image_sketch", images=[image, sketch])  # type: ignore

    #     k = 128
    #     fig = tsne(output["sketch_emb"][:k], output["label"][:k])
    #     self.logger.log_metrics({"val/chart": fig})  # type: ignore

    #     self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        _, loss = self.model_step(batch)
        self.log("test/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.hparams["optimizer"](params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams["scheduler"](optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


class SiameseTester(Siamese):
    def __init__(self, *args, **kwargs):
        super.__init__(args, kwargs)
        self.decoder.eval()
        self.index = faiss.IndexFlatL2()

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            out = self.decoder(batch["image"])
        self.index.add(out)
