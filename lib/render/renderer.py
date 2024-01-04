import numpy as np
import torch
import trimesh
from lightning import LightningModule
from skimage.measure import marching_cubes
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.models.deepsdf import DeepSDF


class DeepSDFRender(LightningModule):
    def __init__(
        self,
        ckpt_path: str,
        reg_loss: bool = True,
        reg_weight: float = 1e-4,
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

    def forward(self, points, mask=None):
        latent = self.latent.expand(points.shape[0], -1)
        if mask is not None:
            return self.model((points[mask], latent[mask])).squeeze()
        return self.model((points, latent)).squeeze()

    def training_step(self, batch, batch_idx):
        torch.autograd.set_detect_anomaly(True)

        # calculate the normals map
        gt_image = batch["gt_image"].squeeze()
        normals, surface_mask = self.render_normals(
            points=batch["points"].squeeze(),
            mask=batch["mask"].squeeze(),
            rays=batch["rays"].squeeze(),
        )
        image = self.to_image(normals, surface_mask)

        # calculate the loss for the object and usefull information to wandb
        loss = torch.nn.functional.mse_loss(image, gt_image)
        self.log("optimize/obj_loss", loss, on_step=True, on_epoch=True)
        self.log("optimize/latent_norm", torch.linalg.norm(self.latent, dim=-1))
        self.logger.log_image("image", [image.detach().cpu().numpy()])
        self.logger.log_image("gt_image", [batch["gt_image"].detach().cpu().numpy()])

        # add prior loss
        reg_loss = None
        if self.hparams["reg_loss"]:
            reg_loss = self.hparams["reg_weight"] * torch.linalg.norm(
                self.latent, dim=-1
            )
            loss += reg_loss
            self.log("optimize/reg_loss", reg_loss, on_step=True, on_epoch=True)

        # log the full loss
        self.log("optimize/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = self.hparams["optimizer"]([self.latent])
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def to_image(self, x, mask=None, default=1):
        resolution = self.hparams["resolution"]
        if mask is not None:
            x[~mask] = default
        x = x.view(resolution, resolution, -1)
        x = (x + 1) / 2
        return x.permute(1, 0, 2)

    def sphere_tracing(
        self,
        points: torch.Tensor,
        rays: torch.Tensor,
        mask: torch.Tensor,
    ):
        device = self.model.device
        # points = points.clone()
        # mask = mask.clone()

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

    def render_normals(
        self,
        points: torch.Tensor,
        rays: torch.Tensor,
        mask: torch.Tensor,
    ):
        points, surface_mask = self.sphere_tracing(points=points, rays=rays, mask=mask)
        points.requires_grad = True
        sdf = self.forward(points, surface_mask)
        # (normals,) = torch.autograd.grad(
        #     outputs=sdf.expand(3, points.shape[0]).T,
        #     inputs=points,
        #     grad_outputs=torch.ones_like(points),
        #     create_graph=True,
        #     retain_graph=True,
        # )
        (normals,) = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True,
        )
        # normals.requires_grad = True
        normals = torch.nn.functional.normalize(normals, dim=-1)
        return normals, surface_mask

    def to_mesh(self, resolution: int = 256, chunk_size: int = 65536):
        min_val, max_val = -1, 1
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
