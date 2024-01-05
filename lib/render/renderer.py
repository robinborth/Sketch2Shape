import numpy as np
import torch
import trimesh
from lightning import LightningModule
from skimage.measure import marching_cubes
from torch.nn.functional import l1_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.models.deepsdf import DeepSDF


class DeepSDFRender(LightningModule):
    def __init__(
        self,
        ckpt_path: str,
        reg_loss: bool = True,
        reg_weight: float = 1e-4,
        image_weight: float = 1,
        prior_idx: int = -1,
        resolution: int = 256,
        n_render_steps: int = 100,
        clamp_sdf: float = 0.1,
        step_scale: float = 1.5,
        surface_eps: float = 1e-03,
        sphere_eps: float = 3e-2,
        optimizer=None,
        scheduler=None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # render options
        self.resolution = resolution
        self.n_render_steps = n_render_steps
        self.clamp_sdf = clamp_sdf
        self.step_scale = step_scale
        self.surface_eps = surface_eps
        self.sphere_eps = sphere_eps
        self.min_val = -1
        self.max_val = 1
        self.log_images = True

        # init model
        self.model = DeepSDF.load_from_checkpoint(self.hparams["ckpt_path"])
        self.model.freeze()
        self.model.eval()

        # init latent either by using a pretrained one ore the mean of the pretrained
        if self.hparams["prior_idx"] >= 0:
            idx = torch.tensor([self.hparams["prior_idx"]])
            latent = self.model.lat_vecs(idx.to(self.model.device)).squeeze()
        else:
            mean = self.model.lat_vecs.weight.mean(0)
            std = self.model.lat_vecs.weight.std(0)
            latent = torch.normal(mean, std)
        self.register_buffer("latent", latent)
        self.latent.requires_grad = True
        self.model.lat_vecs = None

    def log_image(self, key: str, image: torch.Tensor):
        if self.log_images:
            self.logger.log_image(key, [image.detach().cpu().numpy()])  # type: ignore

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

    def sphere_tracing(
        self,
        points: torch.Tensor,
        rays: torch.Tensor,
        mask: torch.Tensor,
    ):
        device = self.model.device
        points = points.clone()
        mask = mask.clone()

        total_points = (points.shape[0],)
        depth = torch.zeros(total_points).to(device)
        sdf = torch.ones(total_points).to(device)

        # sphere tracing
        for _ in range(self.n_render_steps):
            with torch.no_grad():
                sdf_out = self.forward(points=points, mask=mask).to(points)

            sdf_out = torch.clamp(sdf_out, -self.clamp_sdf, self.clamp_sdf)
            depth[mask] += sdf_out * self.step_scale
            sdf[mask] = sdf_out * self.step_scale

            surface_idx = torch.abs(sdf) < self.surface_eps
            # TODO there are holes in the rendering
            # void_idx = points.norm(dim=-1) > 1
            void_idx = depth > 2.0
            mask[surface_idx | void_idx] = False

            points[mask] = points[mask] + sdf[mask, None] * rays[mask]

            if not mask.sum():
                break

        surface_mask = sdf < self.surface_eps
        return points, surface_mask

    def sphere_tracing_min_sdf(
        self,
        points: torch.Tensor,
        rays: torch.Tensor,
        mask: torch.Tensor,
    ):
        device = self.model.device
        points = points.clone()
        mask = mask.clone()
        total_points = (points.shape[0],)
        depth = torch.zeros(total_points).to(device)
        sdf = torch.ones(total_points).to(device)

        min_points = points.clone()
        min_sdf = sdf.clone()

        # sphere tracing
        for _ in range(self.n_render_steps):
            with torch.no_grad():
                sdf_out = self.forward(points=points, mask=mask).to(points)

            sdf_out = torch.clamp(sdf_out, -self.clamp_sdf, self.clamp_sdf)
            depth[mask] += sdf_out * self.step_scale
            sdf[mask] = sdf_out * self.step_scale

            surface_idx = torch.abs(sdf) < self.surface_eps
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

        surface_mask = sdf < self.surface_eps
        return min_points, min_sdf, surface_mask

    def render_normals(
        self,
        points: torch.Tensor,
        mask: torch.Tensor,
    ):
        points.requires_grad = True
        sdf = self.forward(points, mask=mask)
        (normals,) = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True,
        )
        return torch.nn.functional.normalize(normals, dim=-1)

    def forward(self, points, mask=None):
        latent = self.latent.expand(points.shape[0], -1)
        if mask is not None:
            return self.model((points[mask], latent[mask])).squeeze()
        return self.model((points, latent)).squeeze()

    def configure_optimizers(self):
        optimizer = self.hparams["optimizer"]([self.latent])
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def to_mesh(self, resolution: int = 256, chunk_size: int = 65536):
        min_val, max_val = self.min_val, self.max_val
        grid_vals = torch.linspace(min_val, max_val, resolution)
        xs, ys, zs = torch.meshgrid(grid_vals, grid_vals, grid_vals)
        points = torch.stack((xs.ravel(), ys.ravel(), zs.ravel())).transpose(1, 0)

        loader = DataLoader(points, batch_size=chunk_size)  # type: ignore
        sd = []
        for points in tqdm(iter(loader), total=len(loader)):
            points = points.to(self.model.device)
            sd_out = self.forward(points).detach().cpu().numpy()
            sd.append(sd_out)
        sd_cube = np.concatenate(sd).reshape(resolution, resolution, resolution)

        verts, faces, _, _ = marching_cubes(sd_cube, level=0.0)
        verts = verts * ((max_val - min_val) / resolution) + min_val
        return trimesh.Trimesh(vertices=verts, faces=faces)


class DeepSDFNormalRender(DeepSDFRender):
    def training_step(self, batch, batch_idx):
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
        # get the gt image and normals
        gt_image = batch["gt_image"].squeeze()
        gt_surface_mask = batch["gt_surface_mask"].reshape(-1)
        gt_normals = self.image_to_normal(gt_image)
        unit_sphere_mask = batch["mask"].squeeze()

        # calculate the normals map
        min_points, _, surface_mask = self.sphere_tracing_min_sdf(
            points=batch["points"].squeeze(),
            mask=unit_sphere_mask,
            rays=batch["rays"].squeeze(),
        )
        normals = self.render_normals(points=min_points, mask=surface_mask)
        image = self.normal_to_image(normals, surface_mask)
        mask = gt_surface_mask & surface_mask

        # calculate the loss for the object and usefull information to wandb
        normal_loss = l1_loss(gt_normals[mask], normals[mask])
        normal_loss *= self.hparams["image_weight"]
        self.log("optimize/normal_loss", normal_loss, on_step=True, on_epoch=True)

        min_points.requires_grad = True
        min_sdf = torch.abs(self.forward(min_points).to(min_points))
        soft_silhouette = min_sdf - self.surface_eps
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
