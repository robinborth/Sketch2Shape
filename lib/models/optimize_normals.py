import torch
from torch.nn.functional import l1_loss

from lib.models.optimize_latent import LatentOptimizer


class DeepSDFNormalRender(LatentOptimizer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model.lat_vecs = None

    def training_step(self, batch, batch_idx):
        normal = batch["normal"].squeeze(0)  # dim (H, W, 3) and values are (0, 1)
        surface_mask = batch["gt_surface_mask"].squeeze(0)  # dim (H*W, 3)

        # calculate the normals map and embedding
        points, rendered_surface_mask = self.sphere_tracing(
            points=batch["points"].squeeze(),
            rays=batch["rays"].squeeze(),
            mask=batch["mask"].squeeze(),
        )  # dim (H*W, 3), dim (H*W, 3)
        rendered_normal = self.render_normals(
            points=points,
            mask=surface_mask,
        )  # (H, W, 3)

        # calculate the loss
        mask = surface_mask & rendered_surface_mask  # (H*W, 3)
        normal_loss = l1_loss(
            input=normal.reshape(-1, 3)[mask],
            target=rendered_normal.reshape(-1, 3)[mask],
        )
        self.log("optimize/normal_loss", normal_loss)

        # add regularization loss
        reg_loss = torch.tensor(0).to(normal_loss)
        if self.hparams["reg_loss"]:
            reg_loss = torch.norm(self.latent, dim=-1).clone()
            reg_loss *= self.hparams["reg_weight"]
            self.log("optimize/reg_loss", reg_loss)

        loss = reg_loss + normal_loss
        self.log("optimize/loss", loss)

        # visualize the different images
        self.log_image("normal", normal)
        self.log_image("rendered_normal", rendered_normal)

        return loss
