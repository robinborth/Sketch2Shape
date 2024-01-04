import torch
from lightning import LightningModule
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class Siamese(LightningModule):
    def __init__(
        self,
        decoder: torch.nn.Module,
        miner: miners.BaseMiner,
        loss: losses.BaseMetricLossFunction,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        mine_full_batch: bool = False,
        scale_loss: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.decoder = decoder
        self.miner = miner
        self.loss = loss

    def forward(self, batch):
        batch_size = batch["sketch"].shape[0]
        decoder_input = torch.concatenate([batch["sketch"], batch["image"]])
        decoder_emb = self.decoder(decoder_input)
        sketch_emb = decoder_emb[:batch_size]
        image_emb = decoder_emb[batch_size:]
        return {"sketch_emb": sketch_emb, "image_emb": image_emb}

    def model_step(self, batch):
        output = self.forward(batch)
        if self.hparams["mine_full_batch"]:
            embeddings = torch.concatenate([output["sketch_emb"], output["image_emb"]])
            labels = torch.concatenate([batch["label"], batch["label"]])
            miner_output = self.miner(embeddings=embeddings, labels=labels)
            loss = self.loss(
                embeddings=embeddings,
                labels=labels,
                indices_tuple=miner_output,
            )
        else:
            labels = batch["label"]
            miner_output = self.miner(
                embeddings=output["sketch_emb"],
                labels=labels,
                ref_emb=output["image_emb"],
            )
            loss = self.loss(
                embeddings=output["sketch_emb"],
                labels=labels,
                indices_tuple=miner_output,
                ref_emb=output["image_emb"],
            )

        output["miner_count"] = len(miner_output[0])
        output["miner_max_count"] = len(lmu.get_all_triplets_indices(labels)[0])
        output["miner_ratio"] = output["miner_count"] / output["miner_max_count"]

        if self.hparams["scale_loss"]:
            loss *= output["miner_ratio"]

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
        return loss

    def test_step(self, batch, batch_idx):
        _, loss = self.model_step(batch)
        self.log("test/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.hparams["optimizer"](params=self.parameters())
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
