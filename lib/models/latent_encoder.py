import torch
from lightning import LightningModule
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights


class LatentEncoder(LightningModule):
    def __init__(
        self,
        embedding_size: int = 256,
        pretrained: bool = True,
        lr_head: float = 1e-03,
        lr_backbone: float = 1e-05,
        scheduler=None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.lr_head = lr_head
        self.lr_backbone = lr_backbone

        # load the resnet18 backbone
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = torch.nn.Identity()
        self.head = torch.nn.Linear(in_features=512, out_features=embedding_size)

    def forward(self, batch):
        x = self.backbone(batch)
        return self.head(x)

    def model_step(self, batch, split: str = "train"):
        emb = self.forward(batch["image"])  # (B, D)
        loss = torch.nn.functional.l1_loss(emb, batch["latent"])
        self.log(f"{split}/loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.model_step(batch, split="train")

    def validation_step(self, batch, batch_idx):
        return self.model_step(batch, split="val")

    def test_step(self, batch, batch_idx):
        return self.model_step(batch, split="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.backbone.parameters(), "lr": self.lr_backbone},
                {"params": self.head.parameters(), "lr": self.lr_head},
            ]
        )
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "train/loss"},
            }
        return {"optimizer": optimizer}
