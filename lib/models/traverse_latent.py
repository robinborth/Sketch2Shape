import torch

from lib.data.metainfo import MetaInfo
from lib.models.optimize_latent import LatentOptimizer


class DeepSDFLatentTraversal(LatentOptimizer):
    def __init__(
        self,
        prior_idx_start: int = -1,
        prior_idx_end: int = -1,
        create_mesh: bool = True,
        create_video: bool = True,
        # settings to visualize the snn loss
        compute_snn_loss: bool = False,
        data_dir: str = "/data",
        siamese_ckpt_path: str = "/ckpt_path",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.meshes: list[dict] = []

        latent_start = self.model.get_latent(prior_idx_start)
        self.register_buffer("latent_start", latent_start)

        latent_end = self.model.get_latent(prior_idx_end)
        self.register_buffer("latent_end", latent_end)

        self.model.lat_vecs = None

        self.compute_snn_loss = compute_snn_loss
        if compute_snn_loss:
            self.metainfo = MetaInfo(data_dir)
            # TODO think about how to load the image
            sketch = self.metainfo.load_image(prior_idx_end, 11, 0)
            self.register_buffer("sketch", sketch)
            ...

    def on_validation_start(self) -> None:
        if self.compute_snn_loss:
            self.siamese.eval()
            self.sketch_emb = self.siamese(self.sketch)
            sketch_norm = torch.norm(self.sketch_emb, dim=-1)
            self.log("optimize/sketch_norm", sketch_norm, on_step=True)
            self.log_image("sketch", self.sketch)

    def validation_step(self, batch, batch_idx):
        t = batch[0]  # t = [0, 1]
        self.latent = (1 - t) * self.latent_start + t * self.latent_end  # interpolate

        if self.hparams["create_mesh"]:
            mesh = self.to_mesh()
            self.meshes.append(mesh)

        if self.hparams["create_video"]:
            self.capture_camera_frame()

        if self.compute_snn_loss:
            rendered_normals = self.capture_camera_frame()
            normals = rendered_normals.permute(2, 0, 1)[None, ...]
            normal_emb = self.siamese(normals)
            normal_norm = torch.norm(normal_emb, dim=-1)
            self.log("optimize/normal_norm", normal_norm, on_step=True)

            loss = torch.norm(self.sketch_emb - normal_emb, dim=-1)
            self.log("optimize/siamese_loss", loss, on_step=True)
