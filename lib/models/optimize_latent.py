import open3d as o3d
import torch
from lightning import LightningModule

from lib.eval.chamfer_distance import ChamferDistanceMetric
from lib.eval.clip_score import CLIPScoreMetric
from lib.eval.earth_movers_distance import EarthMoversDistanceMetric
from lib.eval.frechet_inception_distance import FrechetInceptionDistanceMetric
from lib.models.deepsdf import DeepSDF


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
        # image logger settings
        log_images: bool = True,
        capture_rate: int = 30,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # init model
        self.model = DeepSDF.load_from_checkpoint(
            ckpt_path,
            strict=False,
            # mesh settings
            mesh_resolution=mesh_resolution,
            mesh_chunk_size=mesh_chunk_size,
            # rendering settings
            n_render_steps=n_render_steps,
            clamp_sdf=clamp_sdf,
            step_scale=step_scale,
            surface_eps=surface_eps,
            sphere_eps=sphere_eps,
            normal_eps=normal_eps,
        )
        self.model.freeze()
        self.model.create_camera()

        # init latent either by pretrained, mean or random
        latent = self.model.get_latent(prior_idx)
        self.register_buffer("latent", latent)

        # mesh
        self.mesh: o3d.geometry.TriangleMesh = None

        #  metrics
        self.chamfer_distance = ChamferDistanceMetric()
        # TODO add the other metrics here

    def forward(self, points: torch.Tensor, mask=None):
        return self.model(points=points, latent=self.latent, mask=mask)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Please provide the optimization implementation.")

    def on_train_epoch_start(self):
        self.model.eval()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx % self.hparams["capture_rate"] == 0:
            self.capture_camera_frame()

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

    def to_mesh(self) -> o3d.geometry.TriangleMesh:
        self.mesh = self.model.to_mesh(self.latent)
        return self.mesh

    def capture_camera_frame(self):
        image = self.model.capture_camera_frame(self.latent)
        self.model.log_image("video_frame", image)
        return image
