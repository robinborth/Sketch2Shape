import torch
from lightning import LightningModule
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights


class Siamese(LightningModule):
    def __init__(
        self,
        margin: float = 0.2,
        embedding_size: int = 128,
        pretrained: bool = True,
        reg_loss: bool = True,
        reg_weight: float = 1e-03,
        optimizer=None,
        scheduler=None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # load the resnet18 backbone
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.decoder = resnet18(weights=weights)
        self.decoder.fc = torch.nn.Linear(in_features=512, out_features=embedding_size)

    def forward(self, batch):
        return self.decoder(batch)

    def get_all_triplets_indices(self, labels):
        labels1 = labels.unsqueeze(1)
        labels2 = labels.unsqueeze(0)
        matches = (labels1 == labels2).byte()
        diffs = matches ^ 1
        matches.fill_diagonal_(0)
        triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
        return torch.where(triplets)

    def model_step(self, batch, split: str = "train"):
        emb = self.forward(batch["image"])  # (B, D)

        # calculate the anchor, positive, negative indx
        a_idx, p_idx, n_idx = self.get_all_triplets_indices(batch["label"])

        # calculate the triplet loss
        m = self.hparams["margin"]
        d_ap = torch.norm(emb[a_idx] - emb[p_idx], dim=-1)  # l2_dist
        self.log(f"{split}/distance_anchor_positive", d_ap.mean(), prog_bar=True)
        d_an = torch.norm(emb[a_idx] - emb[n_idx], dim=-1)  # l2_dist
        self.log(f"{split}/distance_anchor_negative", d_an.mean(), prog_bar=True)
        triplet_loss = torch.relu(d_ap - d_an + m)  # max(0, d_ap - d_an + m)
        triplet_mask = triplet_loss > 0
        triplet_loss = triplet_loss[triplet_mask].mean()  # no zero avg
        # triplet_loss = triplet_loss.mean()  # full avg
        self.log(f"{split}/triplet_loss", triplet_loss, prog_bar=True)

        triplet_count = triplet_mask.sum().float()
        self.log(f"{split}/triplet_count", triplet_count, prog_bar=True)

        # calculate the reg loss based on the embeeddings
        reg_loss = torch.tensor(0).to(triplet_loss)
        if self.hparams["reg_loss"]:
            reg_loss = torch.norm(emb, dim=-1).mean()  # l2_reg on embedding
            reg_loss *= self.hparams["reg_weight"]
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
