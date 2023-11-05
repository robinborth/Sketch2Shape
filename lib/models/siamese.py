import lightning as L
import torch
from omegaconf import DictConfig
from pytorch_metric_learning import losses, miners, reducers

from lib.models.layers import SimpleDecoder


class Siamese(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # note that we share the weights of the decoder for sketches and images
        self.decoder = SimpleDecoder(
            image_size=cfg.image_size,
            embedding_size=cfg.embedding_size,
        )
        self.miner = miners.TripletMarginMiner(
            margin=cfg.margin,
            type_of_triplets=cfg.type_of_triplets,
        )
        self.loss = losses.TripletMarginLoss(margin=cfg.margin)

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
        self.log("train/loss", loss, prog_bar=True, batch_size=self.cfg.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
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
        self.log("val/loss", loss, prog_bar=True, batch_size=self.cfg.batch_size)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.cfg.learning_rate,
        )
