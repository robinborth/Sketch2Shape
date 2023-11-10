import torch
import torch.nn as nn
from lightning import LightningModule
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig
from pytorch_metric_learning import losses, miners
from torch.optim.adam import Adam

from lib.models.layers import DummyDecoder, SimpleDecoder


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

    def forward(self, batch):
        sketch_emb = self.decoder(batch["sketch"])
        image_emb = self.decoder(batch["image"])
        return {"sketch_emb": sketch_emb, "image_emb": image_emb}

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        miner_output = self.miner(
            embeddings=output["sketch_emb"],
            labels=batch["label"],
            ref_emb=output["image_emb"],
            ref_labels=batch["label"],
        )
        loss = self.loss(
            embeddings=output["sketch_emb"],
            labels=batch["label"],
            indices_tuple=miner_output,
            ref_emb=output["image_emb"],
            ref_labels=batch["label"],
        )
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
        )
        return loss

    # def validation_step(self, batch, batch_idx):
    #     output = self.forward(batch)
    #     miner_output = self.miner(
    #         embeddings=output["sketch_emb"],
    #         labels=batch["label"],
    #         ref_emb=output["image_emb"],
    #         ref_labels=batch["label"],
    #     )
    #     loss = self.loss(
    #         embeddings=output["sketch_emb"],
    #         labels=batch["label"],
    #         indices_tuple=miner_output,
    #         ref_emb=output["image_emb"],
    #         ref_labels=batch["label"],
    #     )
    #     self.log("val/loss", loss, prog_bar=True, batch_size=self.cfg.batch_size)
    #     return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",  # TODO change to val/loss
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
