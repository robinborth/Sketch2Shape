import torch

from lib.models.loss import Loss


class LatentEncoder(Loss):
    def __init__(
        self,
        support_latent: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    def model_step(self, batch, batch_idx: int, split: str = "train"):
        emb = self.forward(batch["image"], batch["type_idx"])  # (B, D)
        loss = torch.nn.functional.l1_loss(emb, batch["latent"])
        self.log(f"{split}/loss", loss, prog_bar=True)

        # log one image from the training
        type_idx, image = batch["type_idx"][0], batch["image"][0]
        self.log_image(key=f"image_{type_idx}", image=image, batch_idx=batch_idx)

        return loss
