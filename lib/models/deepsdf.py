import math

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from skimage.measure import marching_cubes
from torch.utils.data import DataLoader
from torchvision.transforms import v2
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
        # lightning settings
        ambient: float = 0.2,
        diffuse: float = 0.5,
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

    def create_camera(
        self,
        azim: float = 40.0,
        elev: float = -30.0,
        dist: float = 4.0,
        width: int = 256,
        height: int = 256,
        focal: int = 512,
        sphere_eps: float = 1e-01,
        surface_eps: float = 1e-03,
    ):
        device = self.device
        camera = Camera(
            azim=azim,
            elev=elev,
            dist=dist,
            width=width,
            height=height,
            focal=focal,
            sphere_eps=sphere_eps,
        )
        points, rays, mask = camera.unit_sphere_intersection_rays()
        camera_position = camera.camera_position()
        self.camera_points = torch.tensor(points, device=device)
        self.camera_rays = torch.tensor(rays, device=device)
        self.camera_mask = torch.tensor(mask, dtype=torch.bool, device=device)
        self.camera_position = torch.tensor(camera_position, device=device)
        self.world_to_camera = torch.tensor(camera.get_world_to_camera(), device=device)
        self.camera_width = width
        self.camera_height = height
        self.camera_focal = focal

    def capture_camera_frame(
        self,
        latent: torch.Tensor,
        mode: str = "normal",
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            points, surface_mask = self.sphere_tracing(
                latent=latent,
                points=self.camera_points,
                mask=self.camera_mask,
                rays=self.camera_rays,
            )
            fn = self.render_normals if mode == "normal" else self.render_grayscale
            image = fn(
                points=points,
                latent=latent,
                mask=surface_mask,
            )
        return image

    def log_image(self, key: str, image: torch.Tensor):
        image = image.detach().cpu().numpy()
        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(key, [image])  # type: ignore

    ############################################################
    # Rendering Utils
    ############################################################

    def normal_to_grayscale(
        self,
        normal: torch.Tensor,
        ambient: float = 0.2,
        diffuse: float = 0.5,
        camera_position=None,
    ):
        """Transforms the normal after render_normals to grayscale."

        Args:
            normal (torch.Tensor): The normal image of dim: (H, W, 3) and range (0, 1)

        Returns:
            torch.Tensor: The grayscale image of dim (H, W, 3) and range (0, 1)
        """
        mask = normal.sum(-1) > 2.95
        N = (normal - 0.5) / 0.5
        C = self.camera_position.clone()
        if camera_position is not None:
            C = camera_position.clone()
        L = C / torch.norm(C)
        grayscale = torch.zeros_like(normal)
        grayscale += ambient
        grayscale += diffuse * (N @ L)[..., None]
        grayscale[mask, :] = 1.0
        grayscale = torch.clamp(
            grayscale,
            torch.tensor(0.0, device=grayscale.device),
            torch.tensor(1.0, device=grayscale.device),
        )
        return grayscale

    def image_to_siamese(self, image: torch.Tensor) -> torch.Tensor:
        """Transforms the rendered normal or grayscale to siamese input."

        Args:
            image (torch.Tensor): The image of dim: (H, W, 3) and range (0, 1)

        Returns:
            torch.Tensor: The input for the siamese network of dim (1, 3, H, W)
        """
        image = image.permute(2, 0, 1)  # (3, H, W)
        image = (image - 0.5) / 0.5  # scale from (0, 1) -> (-1, 1) with mean,std=0.5
        return image[None, ...]  # (1, 3, H, W)

    def loss_input_to_image(self, loss_input: torch.Tensor) -> torch.Tensor:
        """Transforms a loss_input to a image that can be plotted."

        Args:
            loss_input (torch.Tensor): The input of dim: (1, 3, H, W); range: (-1, 1)

        Returns:
            torch.Tensor: The transformed image of dim (H, W, 3).
        """
        loss_input = loss_input.squeeze(0)  # (3, H, W)
        assert loss_input.dim() == 3
        loss_input = (loss_input * 0.5) + 0.5  # (-1, 1) -> (0, 1)
        return loss_input.permute(1, 2, 0)  # (H, W, 3)

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
        return self.normal_to_grayscale(
            normal=rendered_normal,
            ambient=self.hparams["ambient"],
            diffuse=self.hparams["diffuse"],
        )

    def render_silhouette(
        self,
        normals: torch.Tensor,
        points: torch.Tensor,
        latent: torch.Tensor,
        silhouette_surface_eps: float = 5e-02,
        proj_blur_kernal_size: int = 5,
        proj_blur_sigma: float = 3.0,
        proj_blur_eps: float = 0.5,
        weight_blur_kernal_size: int = 9,
        weight_blur_sigma: float = 9.0,
        weight_blur_eps: float = 10.0,
        return_full: bool = True,
    ):
        """Converts an image into an weighted silhouette.

        Args:
            normals (torch.Tensor): Normal image of dim (H, W, 3) with range (0, 1)
            points (torch.Tensor): The min points closest to the surface of dim (W*H, 3)
            latent (torch.Tensor): The latent that reconstructs the full surface.
            silhouette_surface_eps (float, optional): The threshold of the cloestest
                points that are used to calculate the surface. Defaults to 5e-02.
        """
        width, height = self.camera_width, self.camera_height

        # calculate the base silhouette from the initial rendering
        min_normals = (normals - 0.5) / 0.5  # (H, W, 3) with range (-1, 1) and norm=1.0
        base_silhouette = min_normals.sum(-1) > 2.95  # (H, W)

        # silhouette with higher threshold, that blows up the rendering
        min_eps = self.hparams["surface_eps"]
        max_eps = silhouette_surface_eps
        min_sdf = self.forward(points, latent=latent)
        min_sdf = torch.abs(min_sdf).reshape(width, height)
        extra_silhouette = (min_sdf < max_eps) & (min_sdf > min_eps)  # (H, W)

        # project the points from the extra silhouette to the surface in world coords
        min_points = points.reshape(width, height, 3)  # (H, W, 3)
        idx = torch.where(extra_silhouette)
        w_points = min_points[idx] - min_normals[idx] * min_sdf[idx][..., None]  # (X,3)
        # transform the points into camera coords
        c_points = torch.ones((w_points.shape[0], 4), dtype=torch.float32)
        c_points[:, :3] = w_points
        c_points = self.world_to_camera @ c_points.T
        c_points = c_points.T
        # convert the points into pixels on the image plane
        focal = self.focal
        pxs = ((width * 0.5) - (c_points[:, 0] * focal) / c_points[:, 2]).to(torch.int)
        pys = ((height * 0.5) - (c_points[:, 1] * focal) / c_points[:, 2]).to(torch.int)
        # filter the points that are not on the image plane after projection
        inside_mask = (pxs >= 0) & (pxs < width) & (pys >= 0) & (pys < height)
        pxs = pxs[inside_mask]
        pys = pys[inside_mask]
        # sum up the pixels on the projected silhouette
        unique_idx, counts = torch.stack([pys, pxs]).unique(dim=1, return_counts=True)
        proj_silhouette = torch.zeros_like(min_sdf)  # (H, W)
        proj_silhouette[unique_idx[0], unique_idx[1]] = counts.to(torch.float32)

        # blur the silhouette and filter noise out
        proj_blur = v2.GaussianBlur(proj_blur_kernal_size, proj_blur_sigma)
        proj_blur_silhouette = proj_blur(torch.stack([proj_silhouette] * 3))[0]
        proj_blur_silhouette = torch.clip(proj_blur_silhouette - proj_blur_eps, min=0)

        # blur the base silhouette to get an density map
        weight_blur = v2.GaussianBlur(weight_blur_kernal_size, weight_blur_sigma)
        base_blur_silhouette = weight_blur(torch.stack([base_silhouette] * 3))[0]

        # weight map of the blured projected silhouette, to encourage creating
        # silhouette in empty space
        weights = torch.clip(-torch.log(base_blur_silhouette), 0, weight_blur_eps)
        weighted_silhouette = proj_blur_silhouette * weights  # (H, W)

        if return_full:
            return {
                "min_sdf": min_sdf,
                "base_silhouette": base_silhouette,
                "extra_silhouette": extra_silhouette,
                "proj_silhouette": proj_silhouette,
                "proj_blur_silhouette": proj_blur_silhouette,
                "base_blur_silhouette": base_blur_silhouette,
                "weighted_silhouette": weighted_silhouette,
            }
        return weighted_silhouette

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

        total_points = (points.shape[0],)
        sdf = torch.ones(total_points, device=self.device)

        points = points.clone()
        mask = mask.clone()

        # track the points closest to the surface
        min_points = points.clone()
        min_sdf = sdf.clone()

        # sphere tracing
        for _ in range(self.hparams["n_render_steps"]):
            # get the deepsdf values from the model
            with torch.no_grad():
                sdf_out = self.forward(
                    points=points,
                    latent=latent,
                    mask=mask,
                ).to(points)

            # transform the sdf value
            sdf_out = torch.clamp(sdf_out, -clamp_sdf, clamp_sdf)
            sdf_out = sdf_out * step_scale

            # update the depth and the sdf values
            sdf[mask] = sdf_out

            # check if the rays converged
            surface_idx = torch.abs(sdf) < surface_eps
            void_idx = points.norm(dim=-1) > (1 + self.hparams["sphere_eps"] * 2)
            mask[surface_idx | void_idx] = False

            # update the current point on the ray
            points[mask] = points[mask] + sdf[mask, None] * rays[mask]

            # update the closest points to the surface
            min_mask = torch.abs(sdf) < torch.abs(min_sdf)
            min_sdf[min_mask] = sdf[min_mask]
            min_points[min_mask] = points[min_mask]

            # check if converged
            if not mask.sum():
                break

        surface_mask = sdf < surface_eps
        return min_points, surface_mask

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
