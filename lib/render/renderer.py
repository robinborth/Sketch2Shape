import torch
from torch.nn.functional import l1_loss, normalize

from lib.models.deepsdf import DeepSDFLatentOptimizerBase
from lib.models.siamese import Siamese


class DeepSDFRenderBase(DeepSDFLatentOptimizerBase):
    def __init__(
        self,
        n_render_steps: int = 100,
        clamp_sdf: float = 0.1,
        step_scale: float = 1.5,
        surface_eps: float = 1e-03,
        sphere_eps: float = 3e-02,
        ambient: float = 0.5,
        diffuse: float = 0.3,
        specular: float = 0.3,
        shininess: float = 200.0,
        log_images: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # self.model.lat_vecs = None

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

    def render_image(
        self,
        points: torch.Tensor,
        mask: torch.Tensor,
        camera_position: torch.Tensor,
        light_position: torch.Tensor,
    ):
        normals = self.render_normals(points=points, mask=mask)

        N = normals
        L = normalize(light_position - points)
        V = normalize(camera_position - points)

        image = torch.zeros_like(N)
        image += self.hparams["ambient"]
        image += self.hparams["diffuse"] * (L * N).sum(dim=-1)[..., None]
        image += self.hparams["specular"] * torch.pow(
            input=(N * normalize(L + V)).sum(dim=-1)[..., None],
            exponent=self.hparams["shininess"] / 4,
        )

        image[~mask] = 1
        return torch.clip(image, 0, 1)

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


class DeepSDFNormalRender(DeepSDFRenderBase):
    def __init__(
        self,
        normal_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

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
        normal_loss *= self.hparams["normal_weight"]
        self.log("optimize/normal_loss", normal_loss)

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
            self.log("optimize/reg_loss", reg_loss)

        # log the full loss
        loss = reg_loss + normal_loss
        self.log("optimize/loss", loss)

        # visualize the different images
        self.log_image("gt_image", gt_image)
        self.log_image("image", image)
        self.log_image("gt_surface_mask", self.to_image(gt_surface_mask))
        self.log_image("surface_mask", self.to_image(surface_mask))
        self.log_image("mask", self.to_image(mask))

        return loss


class DeepSDFNormalSilhouetteRender(DeepSDFRenderBase):
    def __init__(
        self,
        normal_weight: float = 1.0,
        silhouette_weight: float = 1e-03,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

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
        normal_loss *= self.hparams["normal_weight"]
        self.log("optimize/normal_loss", normal_loss)

        # calculate the silhouette loss
        points.requires_grad = True
        min_sdf = torch.abs(self.forward(points).to(points))
        soft_silhouette = min_sdf - self.hparams["surface_eps"]
        silhouette_loss = gt_surface_mask * torch.relu(soft_silhouette)
        silhouette_loss += ~gt_surface_mask * torch.relu(-soft_silhouette)
        silhouette_error_map = silhouette_loss.clone()
        self.log_image("silhouette_error_map", self.to_image(silhouette_error_map))
        silhouette_loss = silhouette_loss[unit_sphere_mask].mean()
        silhouette_loss *= self.hparams["silhouette_weight"]
        self.log("optimize/silhouette_loss", silhouette_loss)

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
            self.log("optimize/reg_loss", reg_loss)

        # log the full loss
        loss = reg_loss + normal_loss + silhouette_loss
        self.log("optimize/loss", loss)

        # visualize the different images
        self.log_image("gt_image", gt_image)
        self.log_image("image", image)
        self.log_image("gt_surface_mask", self.to_image(gt_surface_mask))
        self.log_image("surface_mask", self.to_image(surface_mask))
        self.log_image("mask", self.to_image(mask))

        return loss


class DeepSDFSiameseRender(DeepSDFRenderBase):
    def __init__(
        self,
        siamese_ckpt_path: str = "best.ckpt",
        deepsdf_ckpt_path: str = "best.ckpt",
        image_weight: float = 1.0,
        silhouette_weight: float = 1e-03,
        **kwargs,
    ) -> None:
        super().__init__(ckpt_path=deepsdf_ckpt_path, **kwargs)
        self.siamese = Siamese.load_from_checkpoint(self.hparams["ckpt_path"])
        self.siamese.freeze()

    def training_step(self, batch, batch_idx):
        self.model.eval()
        self.siamese.eval()

        # get the gt image and normals
        gt_image = batch["gt_image"].squeeze()
        gt_surface_mask = batch["gt_surface_mask"].reshape(-1)
        gt_sketch = batch["gt_sketch"].squeeze()
        unit_sphere_mask = batch["mask"].squeeze()

        # calculate the normals map
        points, surface_mask = self.sphere_tracing_min_sdf(
            points=batch["points"].squeeze(),
            mask=unit_sphere_mask,
            rays=batch["rays"].squeeze(),
        )
        rendered_image = self.render_image(
            points=points,
            mask=surface_mask,
            camera_position=batch["camera_position"],
            light_position=batch["light_position"],
        )
        image = self.to_image(rendered_image, surface_mask)
        mask = gt_surface_mask & surface_mask

        # calculate the loss for the object and usefull information to wandb
        siamese_output = self.siamese(dict(sketch=gt_sketch, image=image))
        sketch_emb = normalize(siamese_output["sketch_emb"], dim=-1)
        image_emb = normalize(siamese_output["image_emb"], dim=-1)
        image_loss = -(sketch_emb @ image_emb.T).sum() + 1
        image_loss *= self.hparams["image_weight"]
        self.log("optimize/image_loss", image_loss)

        # calculate the silhouette loss
        points.requires_grad = True
        min_sdf = torch.abs(self.forward(points).to(points))
        soft_silhouette = min_sdf - self.hparams["surface_eps"]
        silhouette_loss = gt_surface_mask * torch.relu(soft_silhouette)
        silhouette_loss += ~gt_surface_mask * torch.relu(-soft_silhouette)
        silhouette_error_map = silhouette_loss.clone()
        self.log_image("silhouette_error_map", self.to_image(silhouette_error_map))
        silhouette_loss = silhouette_loss[unit_sphere_mask].mean()
        silhouette_loss *= self.hparams["silhouette_weight"]
        self.log("optimize/silhouette_loss", silhouette_loss)

        # calculate the loss for the object and usefull information to wandb
        error_map = torch.nn.functional.l1_loss(gt_image, image, reduction="none")
        self.log_image("error_map", error_map)

        # visualize the latent norm
        latent_norm = torch.linalg.norm(self.latent, dim=-1)
        self.log("optimize/latent_norm", latent_norm)

        # add regularization loss
        reg_loss = torch.tensor(0).to(image_loss)
        if self.hparams["reg_loss"]:
            reg_loss = latent_norm * self.hparams["reg_weight"]
            self.log("optimize/reg_loss", reg_loss)

        # log the full loss
        loss = reg_loss + image_loss + silhouette_loss
        self.log("optimize/loss", loss)

        # visualize the different images
        self.log_image("gt_image", gt_image)
        self.log_image("image", image)
        self.log_image("gt_surface_mask", self.to_image(gt_surface_mask))
        self.log_image("surface_mask", self.to_image(surface_mask))
        self.log_image("mask", self.to_image(mask))

        return loss
