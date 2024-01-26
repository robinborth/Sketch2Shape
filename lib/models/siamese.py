import pytorch_metric_learning
import torch
from lightning import LightningModule


class Siamese(LightningModule):
    def __init__(
        self,
        decoder: torch.nn.Module,
        miner: torch.nn.Module,
        regularizer: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        use_regularizer: bool = True,
        use_miner: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.decoder = decoder()
        self.miner = miner()
        self.regularizer = regularizer()
        self.loss = loss()

    def forward(self, batch):
        return self.decoder(batch)

    def model_step(self, batch, split: str = "train"):
        labels = batch["label"]
        embeddings = self.forward(batch["image"])  # (B, D)

        # stats about the mining
        miner_output = self.miner(embeddings=embeddings, labels=labels)
        self.log("train/miner_count", len(miner_output[0]))
        if not self.hparams["use_miner"]:
            miner_output = None

        # get the triplet loss
        triplet_loss = self.loss(embeddings, labels=labels, indices_tuple=miner_output)
        self.log("train/triplet_loss", triplet_loss, prog_bar=True)

        # get the regularize loss
        reg_loss = torch.tensor(0).to(triplet_loss)
        if self.hparams["use_regularizer"]:
            reg_loss = self.regularizer(embeddings)
            self.log(f"{split}/reg_loss", reg_loss, prog_bar=True)

        # compute the final loss
        loss = reg_loss + triplet_loss
        self.log(f"{split}/loss", loss, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.model_step(batch, split="train")

    def validation_step(self, batch, batch_idx):
        return self.model_step(batch, split="val")

    def test_step(self, batch, batch_idx):
        return self.model_step(batch, split="test")

    def configure_optimizers(self):
        optimizer = self.hparams["optimizer"](params=self.parameters())
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "train/loss"},
            }
        return {"optimizer": optimizer}
