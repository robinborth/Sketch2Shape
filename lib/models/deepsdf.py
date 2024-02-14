import math

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from skimage.measure import marching_cubes
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.render.camera import Camera


class DeepSDF(LightningModule):
    def __init__(
        self,
        # learning settings
        decoder_lr: float = 5e-04,
        latents_lr: float = 1e-03,
        reg_loss: bool = True,
        reg_weight: float = 1e-4,
        clamp: bool = True,
        clamp_val: float = 0.1,
        adaptive_sample_strategy: bool = False,
        adaptive_mining_strategy: bool = False,
        scheduler=None,
        # model settings
        latent_size: int = 512,
        num_hidden_layers: int = 8,
        latent_vector_size: int = 256,
        num_latent_vectors: int = 1,
        skip_connection: list[int] = [4],
        weight_norm: bool = False,
        dropout: float = 0.0,
        # mesh settings
        mesh_resolution: int = 128,
        mesh_chunk_size: int = 65536,
        # rendering settings
        n_render_steps: int = 100,
        clamp_sdf: float = 0.1,
        step_scale: float = 1.0,
        surface_eps: float = 1e-03,
        sphere_eps: float = 1e-01,
        normal_eps: float = 5e-03,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # inital layers and first input layer
        layers = []  # type: ignore
        layer = nn.Linear(3 + latent_vector_size, latent_size)
        if weight_norm:
            layer = nn.utils.parametrizations.weight_norm(layer)
        layers.append(nn.Sequential(layer, nn.ReLU()))

        # backbone layers
        for layer_idx in range(2, num_hidden_layers + 1):
            output_size = latent_size
            if layer_idx in skip_connection:
                output_size = latent_size - latent_vector_size - 3
            layer = nn.Linear(latent_size, output_size)
            if weight_norm:
                layer = nn.utils.parametrizations.weight_norm(layer)
            layers.append(nn.Sequential(layer, nn.ReLU(), nn.Dropout(p=dropout)))

        # # output layer and final deepsdf backbone
        layers.append(nn.Sequential(nn.Linear(latent_size, 1), nn.Tanh()))
        self.decoder = nn.Sequential(*layers)

        # latent vectors
        self.lat_vecs = nn.Embedding(num_latent_vectors, latent_vector_size)
        std_lat_vec = 1.0 / np.sqrt(latent_vector_size)
        torch.nn.init.normal_(self.lat_vecs.weight.data, 0.0, std_lat_vec)

    def forward(self, points: torch.Tensor, latent: torch.Tensor, mask=None):
        """The forward pass of the deepsdf model.

        Args:
            points (torch.Tensor): The points of dim (B, N, 3) or (N, 3).
            latent (torch.Tensor): The latent code of dim (B, L) or (L).
            mask (torch.Tensor): The mask before feeding to model of dim or (N). Make
            sure that the dimension of the latent is (L) and of the points (N, 3).

        Returns:
            torch.Tensor: The sdf values of dim (B, N) or (N) or when mask applied the
            dimension depends on the positive entries of the mask hence dim ([0...N]).
        """
        N, L = points.shape[-2], latent.shape[-1]

        if len(latent.shape) == 1:
            latent = latent.unsqueeze(-2).expand(N, L)
        else:
            latent = latent.unsqueeze(-2).expand(-1, N, L)
            assert mask is None  # check that only without batching

        if mask is not None:
            points, latent = points[mask], latent[mask]
        out = torch.cat((points, latent), dim=-1)

        for layer_idx, layer in enumerate(self.decoder):
            if layer_idx in self.hparams["skip_connection"]:
                _skip = torch.cat((points, latent), dim=-1)
                out = torch.cat((out, _skip), dim=-1)
            out = layer(out)

        return out.squeeze(-1)

    def on_train_start(self) -> None:
        self.create_camera()

    def training_step(self, batch, batch_idx):
        gt_sdf = batch["sdf"]  # (B, N)
        points = batch["points"]  # (B, N, 3)
        latent = self.lat_vecs(batch["idx"])  # (B, L)

        sdf = self.forward(points=points, latent=latent)  # (B, N)
        if self.hparams["clamp"]:
            clamp_val = self.hparams["clamp_val"]
            sdf = torch.clamp(sdf, -clamp_val, clamp_val)
            gt_sdf = torch.clamp(gt_sdf, -clamp_val, clamp_val)

        if self.hparams["adaptive_sample_strategy"]:  # curriculum deepsdf
            # values from curriculum deepsdf paper
            eps = 0.025
            if self.current_epoch > 200:
                eps = 0.01
            if self.current_epoch > 600:
                eps = 0.0025
            if self.current_epoch > 1000:
                eps = 0.0
            zero = torch.tensor(0).to(self.device)
            loss = torch.max(torch.abs(gt_sdf - sdf) - eps, zero)
            self.log("train/l1_loss", loss.mean(), on_step=True)
            drop_ratio = (loss == 0.0).sum() / loss.numel()
            self.log("train/drop_ratio", drop_ratio, on_step=True)
        else:  # default deepsdf loss
            loss = torch.nn.functional.l1_loss(sdf, gt_sdf, reduction="none")
            self.log("train/l1_loss", loss.mean(), on_step=True)

        if self.hparams["adaptive_mining_strategy"]:
            # values from curriculum deepsdf paper
            gamma = 0
            if self.current_epoch > 200:
                gamma = 0.1
            if self.current_epoch > 600:
                gamma = 0.2
            if self.current_epoch > 1000:
                gamma = 0.5
            mining_weight = 1 + (gamma * torch.sign(gt_sdf) * torch.sign(gt_sdf - sdf))
            self.log("train/mining_weight", mining_weight.mean(), on_step=True)
            loss *= mining_weight

        reg_loss = torch.tensor(0).to(loss)
        if self.hparams["reg_loss"]:
            reg_loss = torch.norm(latent, dim=-1).clone()
            reg_loss *= min(1, self.current_epoch / 100)
            reg_loss *= self.hparams["reg_weight"]
            self.log("train/reg_loss", reg_loss.mean(), on_step=True)

        final_loss = loss.mean() + reg_loss.mean()
        self.log("train/loss", final_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_wo_reg_loss", loss.mean(), on_step=True)

        return final_loss

    def on_train_epoch_end(self) -> None:
        # train latents
        for prior_idx in range(4):
            latent = self.get_latent(prior_idx)
            image = self.capture_camera_frame(latent)
            self.log_image(f"{prior_idx=}_latent", image)

        # mean latent
        mean_latent = self.get_latent(-1)
        image = self.capture_camera_frame(mean_latent)
        self.log_image("mean_latent", image)

        # random latent
        random_latent = self.get_latent(-2)
        image = self.capture_camera_frame(random_latent)
        self.log_image("random_latent", image)

    def configure_optimizers(self):
        decoder = {
            "params": self.decoder.parameters(),
            "lr": self.hparams["decoder_lr"],
        }
        latents = {
            "params": self.lat_vecs.parameters(),
            "lr": self.hparams["latents_lr"],
        }
        optimizer = torch.optim.Adam([decoder, latents])
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "train/loss"},
            }
        return {"optimizer": optimizer}

    def get_latent(self, prior_idx: int):
        # random latent vector
        if prior_idx == -2:
            mu = self.lat_vecs.weight.mean(0)
            sigma = self.lat_vecs.weight.std(0)
            return torch.randn_like(mu) * sigma + mu
        # mean latent vector
        if prior_idx == -1:
            return self.lat_vecs.weight.mean(0)
        # train latent vector
        idx = torch.tensor([prior_idx]).to(self.device)
        return self.lat_vecs(idx).squeeze()

    ############################################################
    # Logging Utils
    ############################################################

    def create_camera(self, azim: float = 40.0, elev: float = -30.0, dist: float = 4.0):
        camera = Camera(azim=azim, elev=elev, dist=dist)
        points, rays, mask = camera.unit_sphere_intersection_rays()
        self.camera_points = torch.tensor(points, device=self.device)
        self.camera_rays = torch.tensor(rays, device=self.device)
        self.camera_mask = torch.tensor(mask, dtype=torch.bool, device=self.device)

    def capture_camera_frame(self, latent: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            points, surface_mask = self.sphere_tracing(
                latent=latent,
                points=self.camera_points,
                mask=self.camera_mask,
                rays=self.camera_rays,
            )
            normals = self.render_normals(
                points=points,
                latent=latent,
                mask=surface_mask,
            )
        return normals

    def log_image(self, key: str, image: torch.Tensor):
        image = image.detach().cpu().numpy()
        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(key, [image])  # type: ignore

    ############################################################
    # Rendering Utils
    ############################################################

    def normal_to_siamese(self, normal: torch.Tensor) -> torch.Tensor:
        """Transforms the normal after render_normals to siamese input."

        Args:
            normal (torch.Tensor): The normal image of dim: (H, W, 3) and range (0, 1)

        Returns:
            torch.Tensor: The input for the siamese network of dim (1, 3, H, W)
        """
        normal = normal.permute(2, 0, 1)  # (3, H, W)
        normal = (normal - 0.5) / 0.5  # scale from (0, 1) -> (-1, 1) with mean,std=0.5
        return normal[None, ...]  # (1, 3, H, W)

    def siamese_input_to_image(self, siamese_input: torch.Tensor) -> torch.Tensor:
        """Transforms a siamese_input to a image that can be plotted."

        Args:
            siamese_input (torch.Tensor): The input of dim: (1, 3, H, W); range: (-1, 1)

        Returns:
            torch.Tensor: The transformed image of dim (H, W, 3).
        """
        siamese_input = siamese_input.squeeze(0)  # (3, H, W)
        assert siamese_input.dim() == 3
        siamese_input = (siamese_input * 0.5) + 0.5  # (-1, 1) -> (0, 1)
        return siamese_input.permute(1, 2, 0)  # (H, W, 3)

    def render_normals(
        self,
        points: torch.Tensor,
        latent: torch.Tensor,
        mask: torch.Tensor,
    ):
        eps = self.hparams["normal_eps"]
        delta = 1 / (2 * eps)

        inp1 = points + torch.tensor([eps, 0, 0], device=self.device)
        inp2 = points - torch.tensor([eps, 0, 0], device=self.device)
        inp3 = points + torch.tensor([0, eps, 0], device=self.device)
        inp4 = points - torch.tensor([0, eps, 0], device=self.device)
        inp5 = points + torch.tensor([0, 0, eps], device=self.device)
        inp6 = points - torch.tensor([0, 0, eps], device=self.device)

        normal_x = self.forward(inp1, latent, mask) - self.forward(inp2, latent, mask)
        normal_y = self.forward(inp3, latent, mask) - self.forward(inp4, latent, mask)
        normal_z = self.forward(inp5, latent, mask) - self.forward(inp6, latent, mask)
        normals = torch.stack([normal_x, normal_y, normal_z], dim=-1) * delta

        if mask is not None:
            zeros = torch.zeros_like(points, device=self.device)
            zeros[mask] = normals
            normals = zeros

        # convert normals to image
        normals = torch.nn.functional.normalize(normals, dim=-1)
        normals = (normals * 0.5) + 0.5  # from (-1, 1) -> (0, 1) with mean, std = 0.5
        normals[~mask] = 1.0  # white background

        # transform to image
        resolution = int(math.sqrt(points.shape[0]))
        normals = normals.reshape(resolution, resolution, -1)
        return normals

    def render_grayscale(
        self,
        points: torch.Tensor,
        latent: torch.Tensor,
        mask: torch.Tensor,
    ):
        rendered_normal = self.render_normals(points=points, latent=latent, mask=mask)

    ############################################################
    # Sphere Tracing Variants
    ############################################################

    def sphere_tracing(
        self,
        latent: torch.Tensor,
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

        # sphere tracing
        for step in range(self.hparams["n_render_steps"]):
            # get the deepsdf values from the model
            with torch.no_grad():
                sdf_out = self.forward(
                    points=points,
                    latent=latent,
                    mask=mask,
                ).to(points)

            # transform the sdf value
            sdf_out = torch.clamp(sdf_out, -clamp_sdf, clamp_sdf)
            if step > 50:
                sdf_out = sdf_out * 0.5
            sdf_out = sdf_out * step_scale

            # update the depth and the sdf values
            depth[mask] += sdf_out
            sdf[mask] = sdf_out

            # check if the rays converged
            surface_idx = torch.abs(sdf) < surface_eps
            void_idx = points.norm(dim=-1) > (1 + self.hparams["sphere_eps"] * 2)
            mask[surface_idx | void_idx] = False

            # update the current point on the ray
            points[mask] = points[mask] + sdf[mask, None] * rays[mask]

            # check if converged
            if not mask.sum():
                break

        surface_mask = sdf < surface_eps
        return points, surface_mask

    ############################################################
    # Mesh Utils
    ############################################################

    def to_mesh(self, latent: torch.Tensor) -> o3d.geometry.TriangleMesh:
        self.eval()
        resolution = self.hparams["mesh_resolution"]
        chunk_size = self.hparams["mesh_chunk_size"]
        min_val, max_val = -1, 1

        grid_vals = torch.linspace(min_val, max_val, resolution)
        xs, ys, zs = torch.meshgrid(grid_vals, grid_vals, grid_vals, indexing="ij")
        points = torch.stack((xs.ravel(), ys.ravel(), zs.ravel())).transpose(1, 0)

        loader = DataLoader(points, batch_size=chunk_size)  # type: ignore
        sd = []
        for points in tqdm(iter(loader), total=len(loader)):
            points = points.to(self.device)
            sd_out = self.forward(points, latent=latent).detach().cpu().numpy()
            sd.append(sd_out)
        sd_cube = np.concatenate(sd).reshape(resolution, resolution, resolution)

        verts, faces, _, _ = marching_cubes(sd_cube, level=0.0)
        verts = verts * ((max_val - min_val) / resolution) + min_val

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        return mesh
