from typing import Dict

import torch
from lightning.pytorch.utilities import grad_norm
from torch.nn.functional import l1_loss

from lib.models.deepsdf import DeepSDFLatentOptimizerBase
from lib.models.siamese import Siamese
from lib.render.camera import Camera


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
        approx_normal_eps: float = 0.005,
        # default view settings
        default_azim: int = 0,
        default_elev: int = 45,
        default_dist: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model.lat_vecs = None

        self._save_default_view()

    def on_train_epoch_start(self):
        self.model.eval()

    def _save_default_view(self):
        camera = Camera(
            azim=self.hparams["default_azim"],
            elev=-self.hparams["default_elev"],
            dist=self.hparams["default_dist"],
        )
        points, rays, mask = camera.unit_sphere_intersection_rays()
        self.register_buffer("default_points", points)
        self.register_buffer("default_rays", rays)
        self.register_buffer("default_mask", mask)

    # slows down the training process by quite a bit, use for debugging and visualization ONLY
    # on_after_backward as it respects the gradient accumulation
    def on_after_backward(self):
        # calculate the normals map
        with torch.no_grad():
            points, surface_mask = self.sphere_tracing_min_sdf(
                points=self.default_points,
                mask=self.default_mask,
                rays=self.default_rays,
            )
            normals = self.render_normals(points=points, mask=surface_mask)
        image = self.normal_to_image(normals, surface_mask)
        self.log_image("default_image", image)

        self._save_default_view()

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
        # avoid vanishing gradients
        normals = torch.zeros_like(points, device=self.device)

        # TODO batch the forward
        eps = self.hparams["approx_normal_eps"]
        inp1 = points + torch.tensor([eps, 0, 0], device=self.device)
        inp2 = points - torch.tensor([eps, 0, 0], device=self.device)
        inp3 = points + torch.tensor([0, eps, 0], device=self.device)
        inp4 = points - torch.tensor([0, eps, 0], device=self.device)
        inp5 = points + torch.tensor([0, 0, eps], device=self.device)
        inp6 = points - torch.tensor([0, 0, eps], device=self.device)

        out1 = self.forward(inp1, mask=mask)
        out2 = self.forward(inp2, mask=mask)
        out3 = self.forward(inp3, mask=mask)
        out4 = self.forward(inp4, mask=mask)
        out5 = self.forward(inp5, mask=mask)
        out6 = self.forward(inp6, mask=mask)

        normals_ = torch.stack([out1 - out2, out3 - out4, out5 - out6]).T.float()

        normals_ *= 1 / (2 * eps)

        if mask is None:
            normals = normals_
        else:
            normals[mask] = normals_

        return torch.nn.functional.normalize(normals, dim=-1)

    def render_image(
        self,
        points: torch.Tensor,
        mask: torch.Tensor,
        camera_pos: torch.Tensor,
        lightsource: Dict[str, torch.Tensor],
        obj_reflection: Dict[str, torch.Tensor],
    ):
        normals = self.render_normals(points=points, mask=mask)

        N = normals
        L = torch.nn.functional.normalize(lightsource["position"] - points, dim=-1)
        V = torch.nn.functional.normalize(camera_pos - points, dim=-1)

        image = torch.zeros_like(N)
        image += obj_reflection["ambient"]
        image += obj_reflection["diffuse"] * (L * N).sum(dim=-1)[..., None]
        image += obj_reflection["specular"] * torch.pow(
            input=(N * torch.nn.functional.normalize(L + V, dim=-1)).sum(dim=-1)[
                ..., None
            ],
            exponent=obj_reflection["shininess"] / 4,
        )

        image[~mask] = 1
        return torch.clip(image, 0, 1)

    # regular sphere tracing
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

    # TODO remove, as its only used for silhoutte loss
    # sphere tracing + store point closest to surface (important if no surface is it)
    def sphere_tracing_min_sdf(
        self,
        points: torch.Tensor,
        rays: torch.Tensor,
        mask: torch.Tensor,
    ):
        clamp_sdf = self.hparams["clamp_sdf"]
        step_scale = self.hparams["step_scale"]
        surface_eps = self.hparams["surface_eps"]

        points = points.clone()
        mask = mask.clone()

        total_points = (points.shape[0],)
        depth = torch.zeros(total_points, device=self.device)
        sdf = torch.ones(total_points, device=self.device)

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

    # sphere tracing and save closest points from every single ray (no mask)
    def sphere_tracing_min_sdf_all(
        self,
        points: torch.Tensor,
        rays: torch.Tensor,
    ):
        clamp_sdf = self.hparams["clamp_sdf"]
        step_scale = self.hparams["step_scale"]

        points = points.clone()

        total_points = (points.shape[0],)
        depth = torch.zeros(total_points, device=self.device)
        sdf = torch.ones(total_points, device=self.device)

        min_points = points.clone()
        min_sdf = sdf.clone()
        depth_at_min = depth.clone()

        # sphere tracing
        for _ in range(self.hparams["n_render_steps"]):
            with torch.no_grad():
                sdf_out_unclamped = self.forward(points=points, mask=None).to(points)

            sdf_out = torch.clamp(sdf_out_unclamped, -clamp_sdf, clamp_sdf)
            depth += sdf_out * step_scale
            sdf = sdf_out * step_scale

            points = points + sdf[..., None] * rays

            min_mask = torch.abs(sdf_out_unclamped) < torch.abs(min_sdf)
            min_sdf[min_mask] = sdf_out_unclamped[min_mask]
            depth_at_min[min_mask] = depth[min_mask]
            min_points[min_mask] = points[min_mask]

        return min_points, depth_at_min


class DeepSDFNormalRender(DeepSDFRenderBase):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    # def training_step(self, batch, batch_idx):
    #     gt_image = batch["gt_image"].squeeze(0)
    #     gt_surface_mask = batch["gt_surface_mask"].reshape(-1)
    #     gt_normals = self.image_to_normal(gt_image)
    #     unit_sphere_mask = batch["mask"].squeeze(0)

    #     # calculate the normals map
    #     points, surface_mask = self.sphere_tracing_min_sdf(
    #         points=batch["points"].squeeze(0),
    #         mask=unit_sphere_mask,
    #         rays=batch["rays"].squeeze(0),
    #     )
    #     normals = self.render_normals(points=points, mask=surface_mask)
    #     image = self.normal_to_image(normals, surface_mask)
    #     mask = gt_surface_mask & surface_mask

    #     # calculate the loss for the object and usefull information to wandb
    #     normal_loss = l1_loss(gt_normals[mask], normals[mask])

    #     self.log("optimize/normal_loss", normal_loss)

    #     latent_norm = torch.linalg.norm(self.latent, dim=-1)
    #     self.log("optimize/latent_norm", latent_norm)
    #     # add regularization loss
    #     if self.hparams["reg_loss"]:
    #         reg_loss = latent_norm * self.hparams["reg_weight"]
    #         self.log("optimize/reg_loss", reg_loss)

    #     loss = reg_loss + normal_loss
    #     self.log("optimize/loss", loss)

    #     self.log(
    #         "optimize/mem_allocated", torch.cuda.memory_allocated() / 1024**2
    #     )  # convert to MIB

    #     # visualize the different images
    #     self.log_image("gt_image", gt_image)
    #     self.log_image("image", image)

    #     return loss

    def training_step(self, batch, batch_idx):
        gt_image = batch["gt_image"].squeeze(0)
        gt_surface_mask = batch["gt_surface_mask"].reshape(-1)
        gt_normals = self.image_to_normal(gt_image)
        unit_sphere_mask = batch["mask"].squeeze(0)

        # calculate the normals map
        points, depth = self.sphere_tracing_min_sdf_all(
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

        self.log(
            "optimize/mem_allocated", torch.cuda.memory_allocated() / 1024**2
        )  # convert to MIB

        # visualize the different images
        self.log_image("gt_image", gt_image)
        self.log_image("image", image)
        self.log_image(
            "depth",
            depth.reshape(self.hparams["resolution"], self.hparams["resolution"]),
        )

        return loss

    # FOR DEBUGGING PURPOSES
    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norm = self.latent.grad.norm(p=2)
        self.log("optimize/grad_norm", norm)


class DeepSDFSketchRender(DeepSDFRenderBase):
    def __init__(
        self,
        snn_path: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.siamese = Siamese.load_from_checkpoint(snn_path)
        self.siamese.freeze()

    def training_step(self, batch, batch_idx):
        self.siamese.eval()

        # get the gt image and normals
        gt_image = batch["gt_image"].squeeze()
        # TODO what is this line?
        # gt_normals = self.image_to_normal(gt_image)
        unit_sphere_mask = batch["mask"].squeeze()
        camera_pos = batch["camera_pos"].squeeze()
        lightsource = batch["lightsource"]
        obj_reflection = batch["obj_reflection"]

        # calculate the normals map
        points, surface_mask = self.sphere_tracing_min_sdf(
            points=batch["points"].squeeze(),
            mask=unit_sphere_mask,
            rays=batch["rays"].squeeze(),
        )
        normals = self.render_object(
            points=points,
            mask=surface_mask,
            camera_pos=camera_pos,
            lightsource=lightsource,
            obj_reflection=obj_reflection,
        )
        # TODO rename normal_to_image
        image = self.normal_to_image(normals, surface_mask)
        # calculate the loss for the object and usefull information to wandb

        gt_emb = self.siamese.decoder(gt_image.reshape(-1, 3, 256, 256))
        image_emb = self.siamese.decoder(image.reshape(-1, 3, 256, 256))
        image_loss = -torch.nn.functional.cosine_similarity(gt_emb, image_emb, dim=-1)
        self.log("optimize/image_loss", image_loss)

        # visualize the latent norm
        latent_norm = torch.linalg.norm(self.latent, dim=-1)
        self.log("optimize/latent_norm", latent_norm)

        # add regularization loss
        reg_loss = torch.tensor(0).to(image_loss)
        if self.hparams["reg_loss"]:
            reg_loss = latent_norm * self.hparams["reg_weight"]
            self.log("optimize/reg_loss", reg_loss)

        # log the full loss
        loss = reg_loss + image_loss
        self.log("optimize/loss", loss)

        # visualize the different images
        self.log_image("gt_image", gt_image)
        self.log_image("image", image)

        return loss
