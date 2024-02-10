import torch
from torch.nn.functional import l1_loss

from lib.data.scheduler import Coarse2FineScheduler
from lib.models.optimize_latent import LatentOptimizer


class DeepSDFNormalRender(LatentOptimizer):
    def __init__(
        self,
        c2f_scheduler: Coarse2FineScheduler,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.c2f_scheduler = c2f_scheduler
        self.model.lat_vecs = None

    def training_step(self, batch, batch_idx):
        self.c2f_scheduler.update(self.current_epoch)  # set the current epoch
        resolution = self.c2f_scheduler.resolution

        points = self.c2f_scheduler.downsample(batch["points"], reducer="max")
        rays = self.c2f_scheduler.downsample(batch["rays"], reducer="avg")
        mask = self.c2f_scheduler.downsample_mask(batch["mask"])

        gt_image = batch["gt_image"].reshape(1, -1, 3)
        gt_image = self.c2f_scheduler.downsample(gt_image, reducer="max")
        gt_image = gt_image.reshape(resolution, resolution, 3)

        gt_normals = self.image_to_normal(gt_image)
        gt_surface_mask = self.c2f_scheduler.downsample_mask(batch["gt_surface_mask"])

        # calculate the normals map
        points, surface_mask = self.sphere_tracing(
            points=points.squeeze(0),
            rays=rays.squeeze(0),
            mask=mask.squeeze(0),
        )
        normals = self.render_normals(points=points, mask=surface_mask)
        image = self.normal_to_image(normals, surface_mask, resolution=resolution)
        mask = gt_surface_mask & surface_mask

        # calculate the loss for the object and usefull information to wandb
        normal_loss = l1_loss(gt_normals[mask], normals[mask])
        self.log("optimize/normal_loss", normal_loss)

        latent_norm = torch.linalg.norm(self.latent, dim=-1)
        self.log("optimize/latent_norm", latent_norm)

        # add regularization loss
        reg_loss = torch.tensor(0).to(normal_loss)
        if self.hparams["reg_loss"]:
            reg_loss = latent_norm * self.hparams["reg_weight"]
            self.log("optimize/reg_loss", reg_loss)

        loss = reg_loss + normal_loss
        self.log("optimize/loss", loss)

        mem_allocated = torch.cuda.memory_allocated() / 1024**2  # convert to MIB
        self.log("optimize/mem_allocated", mem_allocated)

        # visualize the different images
        self.log_image("normal_image", gt_image)
        self.log_image("rendered_normal_image", image)

        return loss
