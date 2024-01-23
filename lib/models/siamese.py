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
        reg_loss: bool = True,
        reg_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.decoder = decoder()
        self.miner = miner()
        self.loss = loss()

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
            triplet_loss = self.loss(
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
            triplet_loss = self.loss(
                embeddings=output["sketch_emb"],
                labels=labels,
                indices_tuple=miner_output,
                ref_emb=output["image_emb"],
            )

        output["miner_count"] = len(miner_output[0])
        output["miner_max_count"] = len(lmu.get_all_triplets_indices(labels)[0])
        output["miner_ratio"] = output["miner_count"] / output["miner_max_count"]

        if self.hparams["scale_loss"]:
            triplet_loss *= output["miner_ratio"]
        output["triplet_loss"] = triplet_loss

        reg_loss = torch.tensor(0).to(triplet_loss)
        if self.hparams["reg_loss"]:
            embedding = torch.concatenate([output["sketch_emb"], output["image_emb"]])
            reg_loss = torch.linalg.norm(embedding, dim=-1).mean()
            reg_loss *= self.hparams["reg_weight"]
        output["reg_loss"] = reg_loss

        output["loss"] = output["reg_loss"] + output["triplet_loss"]

        return output

    def training_step(self, batch, batch_idx):
        output = self.model_step(batch)
        self.log("train/triplet_loss", output["triplet_loss"], prog_bar=True)
        self.log("train/reg_loss", output["triplet_loss"], prog_bar=True)
        self.log("train/loss", output["loss"], prog_bar=True)
        self.log("train/miner_ratio", output["miner_ratio"], prog_bar=True)
        self.log("train/miner_count", output["miner_count"])
        self.log("train/miner_max_count", output["miner_max_count"]),
        return output["loss"]

    def validation_step(self, batch, batch_idx):
        output = self.model_step(batch)
        self.log("val/triplet_loss", output["triplet_loss"], prog_bar=True)
        self.log("val/reg_loss", output["triplet_loss"], prog_bar=True)
        self.log("val/loss", output["loss"], prog_bar=True)
        self.log("val/miner_count", output["miner_count"])
        self.log("val/miner_max_count", output["miner_max_count"]),
        self.log("val/miner_ratio", output["miner_ratio"], prog_bar=True)
        return output["loss"]

    def test_step(self, batch, batch_idx):
        output = self.model_step(batch)
        self.log("test/loss", output["loss"], prog_bar=True)
        return output["loss"]

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
