import torch
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torch.nn.functional import cosine_similarity

from lib.models.resnet import ResNet18


class Loss(LightningModule):
    def __init__(
        self,
        head: str = "linear",
        mode: str = "cosine",
        embedding_size: int = 128,
        pretrained: bool = True,
        shared: bool = False,
        lr_head: float = 1e-03,
        lr_backbone: float = 1e-05,
        support_latent: bool = False,
        scheduler=None,
        # image logger settings
        log_images: bool = True,
        capture_rate: int = 30,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.support_latent = support_latent
        self.shared = shared
        self.mode = mode

        # load the resnet18 backbone
        if shared:
            self.siamese = ResNet18(
                head=head,
                embedding_size=embedding_size,
                pretrained=pretrained,
            )
        else:
            self.tower_0 = ResNet18(
                head=head,
                embedding_size=embedding_size,
                pretrained=pretrained,
            )
            self.tower_1 = ResNet18(
                head=head,
                embedding_size=embedding_size,
                pretrained=pretrained,
            )

    def embedding(self, images, mode: str = "sketch"):
        if mode == "sketch":
            type_idx = torch.zeros((images.shape[0]), dtype=images.device)
        else:
            type_idx = torch.ones((images.shape[0]), dtype=images.device)
        return self.forward(images, type_idx=type_idx)

    def compute(self, emb_0, emb_1):
        """Calculate the loss between two embeddings.

        Args:
            emb_0: The embedding of dim (B, D)
            emb_1: The embeddingd of dim (B, D)

        Returns:
            torch.tensor: The loss value based on a distance or similarity, where lower
            describes that the embeddings are closer.
        """
        if self.hparams["mode"] == "cosine":
            return 1.0 - cosine_similarity(emb_0, emb_1)
        raise NotImplementedError()

    def forward(self, images, type_idx=None):
        """Select the tower or siamese architecture.

        Args:
            images: The images with dim (B, C, W, H)
            type_idx: The index of the tower with dim (B,). Defaults to None.

        Returns:
            torch.Tensor: The embedding of the images of dim (B, D)
        """
        if self.shared:
            assert type_idx is not None
            idx_set = set(type_idx.unique().detach().cpu().numpy())
            assert not idx_set.difference({0, 1})  # check that only 0, 1 is in type_idx
            N, D = images.shape[0], self.hparams["embedding_size"]
            emb = torch.zeros((N, D), device=images.device)
            emb[type_idx == 0] = self.tower_0(images[type_idx == 0])
            emb[type_idx == 1] = self.tower_1(images[type_idx == 1])
            return emb
        return self.siamese(images)

    def get_augmentations_idx(self, labels):
        y_a = torch.argsort(labels)[::2]
        y_b = torch.argsort(labels)[1::2]
        assert len(y_a) == len(y_b)
        return y_a, y_b

    def get_all_triplets_indices(self, labels):
        labels1 = labels.unsqueeze(1)
        labels2 = labels.unsqueeze(0)
        matches = (labels1 == labels2).byte()
        diffs = matches ^ 1
        matches.fill_diagonal_(0)
        triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
        return torch.where(triplets)

    def model_step(self, batch, batch_idx, split: str):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, split="train")

    def validation_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        # configure the optimizer
        lr_backbone = self.hparams["lr_backbone"]
        lr_head = self.hparams["lr_head"]
        if self.shared:
            params = [
                {"params": self.siamese.backbone.parameters(), "lr": lr_backbone},
                {"params": self.siamese.head.parameters(), "lr": lr_head},
            ]
        else:
            params = [
                {"params": self.tower_0.backbone.parameters(), "lr": lr_backbone},
                {"params": self.tower_0.head.parameters(), "lr": lr_head},
                {"params": self.tower_1.backbone.parameters(), "lr": lr_backbone},
                {"params": self.tower_1.head.parameters(), "lr": lr_head},
            ]
        optimizer = torch.optim.Adam(params)

        # configure the scheduler
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "train/loss"},
            }
        return {"optimizer": optimizer}

    def log_image(self, key: str, image: torch.Tensor, batch_idx: int):
        if batch_idx % self.hparams["capture_rate"] == 0:
            image = self.loss_input_to_image(image)
            image = image.detach().cpu().numpy()
            if isinstance(self.logger, WandbLogger):
                self.logger.log_image(key, [image])  # type: ignore

    def loss_input_to_image(self, loss_input: torch.Tensor) -> torch.Tensor:
        """Transforms a loss_input to a image that can be plotted."

        Args:
            loss_input (torch.Tensor): The input of dim: (3, H, W); range: (-1, 1)

        Returns:
            torch.Tensor: The transformed image of dim (H, W, 3).
        """
        assert loss_input.dim() == 3
        loss_input = (loss_input * 0.5) + 0.5  # (-1, 1) -> (0, 1)
        return loss_input.permute(1, 2, 0)  # (H, W, 3)
