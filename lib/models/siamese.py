from collections import defaultdict

import torch
from lightning import LightningModule
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.losses import TripletMarginLoss


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

        self.decoder = decoder
        self.miner = miner
        self.loss = loss

        self.validation_step_outputs = []

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
        m = self.trainer.datamodule.hparams.sampler.keywords["m"]
        max_count = torch.tensor((m - 1) * batch_size * (batch_size - m))
        output["miner_max_count"] = max_count

        # ratio
        output["miner_ratio"] = output["miner_count"] / output["miner_max_count"]

        return output, loss

    def training_step(self, batch, batch_idx):
        output, loss = self.model_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/miner_count", output["miner_count"])
        self.log("train/miner_max_count", output["miner_max_count"]),
        self.log("train/miner_ratio", output["miner_ratio"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output, loss = self.model_step(batch)
        output.update(batch)
        self.log("val/loss", loss, prog_bar=True)
        self.validation_step_outputs.append(output)
        return loss

    def on_validation_end(self) -> None:
        # just combine into one big batch
        output = defaultdict(list)
        for outputs in self.validation_step_outputs:
            for key, value in outputs:
                output[key].append(value)
        for key, value in output.items():
            output[key] = torch.stack(value)  # type: ignore
        output = dict(output)  # type: ignore

        self.validation_step_outputs.clear()

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
