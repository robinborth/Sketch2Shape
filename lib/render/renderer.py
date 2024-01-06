import torch
from torch.nn.functional import l1_loss

from lib.models.deepsdf import DeepSDFLatentOptimizer


class DeepSDFRender(DeepSDFLatentOptimizer):
    def __init__(
        self,
        ckpt_path: str = "best.ckpt",
        prior_idx: int = -1,
        optimizer=None,
        scheduler=None,
        resolution: int = 256,
        reg_loss: bool = True,
        reg_weight: float = 1e-4,
        image_weight: float = 1,
        n_render_steps: int = 100,
        clamp_sdf: float = 0.1,
        step_scale: float = 1.5,
        surface_eps: float = 1e-03,
        sphere_eps: float = 3e-2,
        log_images: bool = True,
    ) -> None:
        super().__init__(
            ckpt_path=ckpt_path,
            prior_idx=prior_idx,
            optimizer=optimizer,
            scheduler=scheduler,
            resolution=resolution,
        )

    def log_image(self, key: str, image: torch.Tensor):
        image = image.detach().cpu().numpy()
        if self.hparams["log_images"]:
            self.logger.log_image(key, [image])  # type: ignore

    def to_image(self, x, mask=None, default=1):
        resolution = self.hparams["resolution"]
        if mask is not None:
            x[~mask] = default
        return x.reshape(resolution, resolution, -1)

    def normal_to_image(self, x, mask=None, default=1):
        x = self.to_image(x=x, mask=mask, default=default)
        return (x - self.min_val) / (self.max_val - self.min_val)

    def image_to_normal(self, x, mask=None, default=1):
        x = x * (self.max_val - self.min_val) + self.min_val
        return x.reshape(-1, 3)

    def render_normals(
        self,
        points: torch.Tensor,
        mask: torch.Tensor,
    ):
        points.requires_grad = True
        sdf = self.forward(points=points, mask=mask)
        (normals,) = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True,
        )
        return torch.nn.functional.normalize(normals, dim=-1)

    def sphere_tracing(
        self,
        points: torch.Tensor,
        rays: torch.Tensor,
        mask: torch.Tensor,
    ):
        device = self.model.device
        clamp_sdf = self.hparams["clamp_sdf"]
        step_scale = self.hparams["step_scale"]
        surface_eps = self.hparams["surface_eps"]

        points = points.clone()
        mask = mask.clone()

        total_points = (points.shape[0],)
        depth = torch.zeros(total_points).to(device)
        sdf = torch.ones(total_points).to(device)

        # sphere tracing
        for _ in range(self.hparams["n_render_steps"]):
            with torch.no_grad():
                sdf_out = self.forward(points=points, mask=mask).to(points)

            sdf_out = torch.clamp(sdf_out, -clamp_sdf, clamp_sdf)
            depth[mask] += sdf_out * step_scale
            sdf[mask] = sdf_out * step_scale

            surface_idx = torch.abs(sdf) < surface_eps
            # TODO there are holes in the rendering
            # void_idx = points.norm(dim=-1) > 1
            void_idx = depth > 2.0
            mask[surface_idx | void_idx] = False

            points[mask] = points[mask] + sdf[mask, None] * rays[mask]

            if not mask.sum():
                break

        surface_mask = sdf < surface_eps
        return points, surface_mask

    def sphere_tracing_min_sdf(
        self,
        points: torch.Tensor,
        rays: torch.Tensor,
        mask: torch.Tensor,
    ):
        device = self.model.device
        clamp_sdf = self.hparams["clamp_sdf"]
        step_scale = self.hparams["step_scale"]
        surface_eps = self.hparams["surface_eps"]

        points = points.clone()
        mask = mask.clone()

        total_points = (points.shape[0],)
        depth = torch.zeros(total_points).to(device)
        sdf = torch.ones(total_points).to(device)

        min_points = points.clone()
        min_sdf = sdf.clone()

        # sphere tracing
        for _ in range(self.hparams["n_render_steps"]):
            with torch.no_grad():
                sdf_out = self.forward(points=points, mask=mask).to(points)

            sdf_out = torch.clamp(sdf_out, -clamp_sdf, clamp_sdf)
            depth[mask] += sdf_out * step_scale
            sdf[mask] = sdf_out * step_scale

            surface_idx = torch.abs(sdf) < surface_eps
            # TODO there are holes in the rendering
            # void_idx = points.norm(dim=-1) > 1
            void_idx = depth > 2.0
            mask[surface_idx | void_idx] = False

            points[mask] = points[mask] + sdf[mask, None] * rays[mask]

            min_mask = torch.abs(sdf) < torch.abs(min_sdf)
            min_sdf[min_mask] = sdf[min_mask]
            min_points[min_mask] = points[min_mask]

            if not mask.sum():
                break

        surface_mask = sdf < surface_eps
        return min_points, surface_mask


class DeepSDFNormalRender(DeepSDFRender):
    def training_step(self, batch, batch_idx):
        self.model.eval()

        # get the gt image and normals
        gt_image = batch["gt_image"].squeeze()
        gt_normals = self.image_to_normal(gt_image)

        # calculate the normals map
        gt_surface_mask = batch["gt_surface_mask"].reshape(-1)
        points, surface_mask = self.sphere_tracing(
            points=batch["points"].squeeze(),
            mask=batch["mask"].squeeze(),
            rays=batch["rays"].squeeze(),
        )
        normals = self.render_normals(points=points, mask=surface_mask)
        image = self.normal_to_image(normals, surface_mask)
        mask = gt_surface_mask & surface_mask

        # calculate the loss for the object and usefull information to wandb
        normal_loss = l1_loss(gt_normals[mask], normals[mask])
        normal_loss *= self.hparams["image_weight"]
        self.log("optimize/normal_loss", normal_loss, on_step=True, on_epoch=True)

        # calculate the loss for the object and usefull information to wandb
        error_map = torch.nn.functional.l1_loss(gt_image, image, reduction="none")
        self.log_image("error_map", error_map)

        # visualize the latent norm
        latent_norm = torch.linalg.norm(self.latent, dim=-1)
        self.log("optimize/latent_norm", latent_norm)

        # add regularization loss
        reg_loss = torch.tensor(0).to(normal_loss)
        if self.hparams["reg_loss"]:
            reg_loss = latent_norm * self.hparams["reg_weight"]
            self.log("optimize/reg_loss", reg_loss, on_step=True, on_epoch=True)

        # log the full loss
        loss = reg_loss + normal_loss
        self.log("optimize/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # visualize the different images
        self.log_image("gt_image", gt_image)
        self.log_image("image", image)
        self.log_image("gt_surface_mask", self.to_image(gt_surface_mask))
        self.log_image("surface_mask", self.to_image(surface_mask))
        self.log_image("mask", self.to_image(mask))

        return loss


class DeepSDFSurfaceNormalRender(DeepSDFRender):
    def training_step(self, batch, batch_idx):
        self.model.eval()

        # get the gt image and normals
        gt_image = batch["gt_image"].squeeze()
        gt_surface_mask = batch["gt_surface_mask"].reshape(-1)
        gt_normals = self.image_to_normal(gt_image)
        unit_sphere_mask = batch["mask"].squeeze()

        # calculate the normals map
        points, surface_mask = self.sphere_tracing_min_sdf(
            points=batch["points"].squeeze(),
            mask=unit_sphere_mask,
            rays=batch["rays"].squeeze(),
        )
        normals = self.render_normals(points=points, mask=surface_mask)
        image = self.normal_to_image(normals, surface_mask)
        mask = gt_surface_mask & surface_mask

        # calculate the loss for the object and usefull information to wandb
        normal_loss = l1_loss(gt_normals[mask], normals[mask])
        normal_loss *= self.hparams["image_weight"]
        self.log("optimize/normal_loss", normal_loss, on_step=True, on_epoch=True)

        points.requires_grad = True
        min_sdf = torch.abs(self.forward(points).to(points))
        soft_silhouette = min_sdf - self.hparams["surface_eps"]
        surface_loss = gt_surface_mask * torch.relu(soft_silhouette)
        surface_loss += ~gt_surface_mask * torch.relu(-soft_silhouette)
        surface_error_map = surface_loss.clone()
        self.log_image("surface_error_map", self.to_image(surface_error_map))
        surface_loss = surface_loss[unit_sphere_mask].mean() * 1e-3
        self.log("optimize/surface_loss", surface_loss, on_step=True, on_epoch=True)

        # calculate the loss for the object and usefull information to wandb
        error_map = torch.nn.functional.l1_loss(gt_image, image, reduction="none")
        self.log_image("error_map", error_map)

        # visualize the latent norm
        latent_norm = torch.linalg.norm(self.latent, dim=-1)
        self.log("optimize/latent_norm", latent_norm)

        # add regularization loss
        reg_loss = torch.tensor(0).to(normal_loss)
        if self.hparams["reg_loss"]:
            reg_loss = latent_norm * self.hparams["reg_weight"]
            self.log("optimize/reg_loss", reg_loss, on_step=True, on_epoch=True)

        # log the full loss
        loss = reg_loss + normal_loss + surface_loss
        self.log("optimize/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # visualize the different images
        self.log_image("gt_image", gt_image)
        self.log_image("image", image)
        self.log_image("gt_surface_mask", self.to_image(gt_surface_mask))
        self.log_image("surface_mask", self.to_image(surface_mask))
        self.log_image("mask", self.to_image(mask))

        return loss
