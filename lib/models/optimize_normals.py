import torch
from torch.nn.functional import l1_loss

from lib.models.optimize_latent import LatentOptimizer


class DeepSDFNormalRender(LatentOptimizer):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model.lat_vecs = None
        # self.model.decoder.load_state_dict(
        #     torch.load("/shared/logs/deprecated/decoder.pt")
        # )

    def training_step(self, batch, batch_idx):
        gt_image = batch["gt_image"].squeeze(0)
        gt_surface_mask = batch["gt_surface_mask"].reshape(-1)
        gt_normals = self.image_to_normal(gt_image)
        unit_sphere_mask = batch["mask"].squeeze(0)

        # calculate the normals map
        points, surface_mask = self.sphere_tracing(
            points=batch["points"].squeeze(0),
            rays=batch["rays"].squeeze(0),
            mask=unit_sphere_mask,
        )
        normals = self.render_normals(points=points, mask=surface_mask)
        image = self.normal_to_image(normals, surface_mask)
        mask = gt_surface_mask & surface_mask

        # calculate the loss for the object and usefull information to wandb
        normal_loss = l1_loss(gt_normals[mask], normals[mask])

        self.log("optimize/normal_loss", normal_loss)

        latent_norm = torch.linalg.norm(self.latent, dim=-1)
        self.log("optimize/latent_norm", latent_norm)
        # add regularization loss
        if self.hparams["reg_loss"]:
            reg_loss = latent_norm * self.hparams["reg_weight"]
            self.log("optimize/reg_loss", reg_loss)

        if self.hparams["reg_loss"]:
            loss = reg_loss + normal_loss
        else:
            loss = normal_loss
        self.log("optimize/loss", loss)

        self.log(
            "optimize/mem_allocated", torch.cuda.memory_allocated() / 1024**2
        )  # convert to MIB

        # visualize the different images
        self.log_image("gt_image", gt_image)
        self.log_image("image", image)

        return loss


# TODO IN PROGRESS - calculate a normal everywhere in space
class DeepSDFNormalEverywhereRender(LatentOptimizer):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model.lat_vecs = None

    def training_step(self, batch, batch_idx):
        gt_image = batch["gt_image"].squeeze(0)
        gt_normals = self.image_to_normal(gt_image)

        # calculate the normals map
        points = self.sphere_tracing_min_sdf_all(
            points=batch["points"].squeeze(0),
            rays=batch["rays"].squeeze(0),
        )
        normals = self.render_normals(points=points, mask=None)
        image = self.normal_to_image(normals, None)

        # calculate the loss for the object and usefull information to wandb
        normal_loss = l1_loss(gt_normals, normals)

        self.log("optimize/normal_loss", normal_loss)

        latent_norm = torch.linalg.norm(self.latent, dim=-1)
        self.log("optimize/latent_norm", latent_norm)
        # add regularization loss
        if self.hparams["reg_loss"]:
            reg_loss = latent_norm * self.hparams["reg_weight"]
            self.log("optimize/reg_loss", reg_loss)

        loss = reg_loss + normal_loss
        self.log("optimize/loss", loss)

        # visualize the different images
        self.log_image("gt_image", gt_image)
        self.log_image("image", image)

        return loss
