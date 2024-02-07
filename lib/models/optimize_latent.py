import numpy as np
import open3d as o3d
import torch
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from skimage.measure import marching_cubes
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.eval.chamfer_distance import ChamferDistanceMetric
from lib.eval.clip_score import CLIPScoreMetric
from lib.eval.earth_movers_distance import EarthMoversDistanceMetric
from lib.eval.frechet_inception_distance import FrechetInceptionDistanceMetric
from lib.models.deepsdf import DeepSDF
from lib.render.camera import Camera


class LatentOptimizer(LightningModule):
    def __init__(
        self,
        # latent optimization settings
        ckpt_path: str = "best.ckpt",
        prior_idx: int = -1,  # random(-2), mean(-1), prior(idx)
        reg_loss: bool = True,
        reg_weight: float = 1e-05,
        optimizer=None,
        scheduler=None,
        image_resolution: int = 256,
        mesh_resolution: int = 128,
        mesh_chunk_size: int = 65536,
        # rendering settings
        n_render_steps: int = 100,
        clamp_sdf: float = 0.1,
        step_scale: float = 1.0,
        surface_eps: float = 1e-03,
        sphere_eps: float = 1e-01,  # similar to settings from camera class
        normal_eps: float = 5e-03,
        ambient: float = 0.5,
        diffuse: float = 0.3,
        specular: float = 0.3,
        shininess: float = 200.0,
        # logger settings
        log_images: bool = True,
        # default video settings
        video_capture_rate: int = 30,
        video_azim: float = 40,  # 00011.png
        video_elev: float = -30,
        video_dist: int = 4,
        # evaluation settings
        # TODO
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # init model
        self.model = DeepSDF.load_from_checkpoint(ckpt_path, strict=False)
        self.model.freeze()

        # init latent either by pretrained, mean or random
        latent = self.model.get_latent(prior_idx)
        self.register_buffer("latent", latent)
        self.mesh: o3d.geometry.TriangleMesh = None

        #  metrics
        self.chamfer_distance = ChamferDistanceMetric()
        # TODO add the other metrics here

        # video settings
        camera = Camera(
            azim=self.hparams["video_azim"],
            elev=self.hparams["video_elev"],
            dist=self.hparams["video_dist"],
        )
        points, rays, mask = camera.unit_sphere_intersection_rays()
        self.video_points, self.video_rays, self.video_mask = points, rays, mask

    def forward(self, points: torch.Tensor, mask=None):
        return self.model(points=points, latent=self.latent, mask=mask)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Please provide the optimization implementation.")

    def on_train_epoch_start(self):
        self.model.eval()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx % self.hparams["video_capture_rate"] == 0:
            self.capture_video_frame()

    def test_step(self, batch, batch_idx):
        gt_surface_samples = batch["surface_samples"].detach().cpu().numpy().squeeze()
        mesh = self.to_mesh()
        chamfer = self.chamfer_distance(mesh, gt_surface_samples)
        self.log("val/chamfer", chamfer)

    def on_test_epoch_start(self):
        self.model.eval()

    def on_before_optimizer_step(self, optimizer):
        """
        In order to debug vanishing gradient problems, we compute the 2-norm for each
        layer. If using mixed precision, the gradients are already unscaled here.
        """
        norm = self.latent.grad.norm(p=2)
        self.log("optimize/grad_norm", norm)

    def configure_optimizers(self):
        self.latent = self.latent.detach()
        self.latent.requires_grad = True
        optimizer = self.hparams["optimizer"]([self.latent])
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    ############################################################
    # Mesh Utils
    ############################################################

    def to_mesh(self) -> o3d.geometry.TriangleMesh:
        self.model.eval()
        resolution = self.hparams["mesh_resolution"]
        chunk_size = self.hparams["mesh_chunk_size"]
        min_val, max_val = -1, 1

        # TODO only sample in the unit sphere, the other points should be positive
        grid_vals = torch.linspace(min_val, max_val, resolution)
        xs, ys, zs = torch.meshgrid(grid_vals, grid_vals, grid_vals, indexing="ij")
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

        # override the current mesh
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(verts)
        self.mesh.triangles = o3d.utility.Vector3iVector(faces)

        return self.mesh

    ############################################################
    # Logging Utils
    ############################################################

    def capture_video_frame(self):
        points = torch.tensor(self.video_points).to(self.device)
        mask = torch.tensor(self.video_mask, dtype=torch.bool).to(self.device)
        rays = torch.tensor(self.video_rays).to(self.device)
        with torch.no_grad():
            points, surface_mask = self.sphere_tracing(
                points=points, mask=mask, rays=rays
            )
            normals = self.render_normals(points=points, mask=surface_mask)
        image = self.normal_to_image(normals, surface_mask)
        self.log_image("video_frame", image)
        return image

    def log_image(self, key: str, image: torch.Tensor):
        image = image.detach().cpu().numpy()
        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(key, [image])  # type: ignore

    ############################################################
    # Rendering Utils
    ############################################################

    def normal_to_image(self, x, mask=None, default=1, resolution=None):
        x = self.to_image(x=x, mask=mask, default=default, resolution=resolution)
        return (x + 1) / 2

    def image_to_normal(self, x, mask=None, default=1):
        x = (x * 2) - 1
        return x.reshape(-1, 3)

    def to_image(self, x, mask=None, default=1, resolution=None):
        if resolution is None:
            resolution = self.hparams["image_resolution"]
        if mask is not None:
            x[~mask] = default
        return x.reshape(resolution, resolution, -1)

    def render_normals(
        self,
        points: torch.Tensor,
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

        normal_x = self.forward(inp1, mask=mask) - self.forward(inp2, mask=mask)
        normal_y = self.forward(inp3, mask=mask) - self.forward(inp4, mask=mask)
        normal_z = self.forward(inp5, mask=mask) - self.forward(inp6, mask=mask)
        normals = torch.stack([normal_x, normal_y, normal_z], dim=-1) * delta

        if mask is not None:
            zeros = torch.zeros_like(points, device=self.device)
            zeros[mask] = normals
            normals = zeros

        return torch.nn.functional.normalize(normals, dim=-1)

    ############################################################
    # Sphere Tracing Variants
    ############################################################

    def sphere_tracing(
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

        # sphere tracing
        for step in range(self.hparams["n_render_steps"]):
            # get the deepsdf values from the model
            with torch.no_grad():
                sdf_out = self.forward(points=points, mask=mask).to(points)

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

    def sphere_tracing_min_sdf_all(
        self,
        points: torch.Tensor,
        rays: torch.Tensor,
    ):
        """Sphere tracing and save closest points from every single ray (no mask)."""
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
