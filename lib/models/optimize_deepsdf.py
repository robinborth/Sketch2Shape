import torch

from lib.models.optimize_latent import LatentOptimizer


class DeepSDFLatentOptimizer(LatentOptimizer):
    def __init__(
        self,
        loss: torch.nn.Module,
        clamp: bool = True,
        clamp_val: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model.lat_vecs = None
        self.loss = loss()

    def training_step(self, batch, batch_idx):
        gt_sdf = batch["sdf"].squeeze()  # (N)
        points = batch["points"].squeeze()  # (N, 3)

        sdf = self.forward(points=points)  # (N)

        if self.hparams["clamp"]:
            clamp_val = self.hparams["clamp_val"]
            sdf = torch.clamp(sdf, -clamp_val, clamp_val)
            gt_sdf = torch.clamp(gt_sdf, -clamp_val, clamp_val)

        l1_loss = self.loss(sdf, gt_sdf)
        self.log("train/l1_loss", l1_loss, on_step=True, on_epoch=True)

        reg_loss = torch.tensor(0).to(l1_loss)
        if self.hparams["reg_loss"]:
            reg_loss = torch.linalg.norm(self.latent, dim=-1).mean()
            reg_loss *= min(1, self.current_epoch / 100)
            reg_loss *= self.hparams["reg_weight"]
            self.log("train/reg_loss", reg_loss, on_step=True, on_epoch=True)

        loss = l1_loss + reg_loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss
