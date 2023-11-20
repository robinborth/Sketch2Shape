import torch
from lightning import LightningModule
from pytorch_metric_learning import losses, miners


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

    def model_step(self, batch):
        output = self.forward(batch)
        # miner_output = self.miner(
        #     embeddings=output["sketch_emb"],
        #     labels=batch["label"],
        #     ref_emb=output["image_emb"],
        #     ref_labels=batch["label"],
        # )
        loss = self.loss(
            embeddings=output["sketch_emb"],
            labels=batch["label"],
            # indices_tuple=miner_output,
            ref_emb=output["image_emb"],
            ref_labels=batch["label"],
        )
        return output, loss

    def training_step(self, batch, batch_idx):
        _, loss = self.model_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, loss = self.model_step(batch)
        self.log("val/loss", loss, prog_bar=True)
        return loss

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
