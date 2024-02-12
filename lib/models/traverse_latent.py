import torch

from lib.data.metainfo import MetaInfo
from lib.data.transforms import SiameseTransform
from lib.models.optimize_latent import LatentOptimizer
from lib.models.siamese import Siamese


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

        latent_start = self.deepsdf.get_latent(prior_idx_start)
        self.register_buffer("latent_start", latent_start)

        latent_end = self.deepsdf.get_latent(prior_idx_end)
        self.register_buffer("latent_end", latent_end)

        self.compute_snn_loss = compute_snn_loss
        if compute_snn_loss:
            # load sketch
            self.metainfo = MetaInfo(data_dir)
            transform = SiameseTransform(mean=0.5, std=0.5)
            sketch = self.metainfo.load_image(prior_idx_end, 11, 0)
            sketch = transform(sketch)[None, ...]
            self.register_buffer("sketch", sketch)  # (3, H, W)
            # load snn
            self.siamese = Siamese.load_from_checkpoint(siamese_ckpt_path)
            self.siamese.freeze()

    def on_validation_start(self) -> None:
        if self.compute_snn_loss:
            self.siamese.eval()
            self.sketch_emb = self.siamese(self.sketch)
            sketch_norm = torch.norm(self.sketch_emb, dim=-1)
            self.log("optimize/sketch_norm", sketch_norm)

    def validation_step(self, batch, batch_idx):
        t = batch[0]  # t = [0, 1]
        self.latent = (1 - t) * self.latent_start + t * self.latent_end  # interpolate

        if self.hparams["create_mesh"]:
            mesh = self.to_mesh()
            self.meshes.append(mesh)

        if self.hparams["create_video"]:
            self.capture_camera_frame()

        if self.compute_snn_loss:
            # calculate the normal embedding
            rendered_normal = self.capture_camera_frame()  # (H, W, 3)
            normal = self.deepsdf.normal_to_siamese(rendered_normal)  # (1, 3, H, W)
            normal_emb = self.siamese(normal)
            normal_norm = torch.norm(normal_emb, dim=-1)
            self.log("optimize/normal_norm", normal_norm, on_step=True)

            # calculate the siamese loss
            loss = torch.norm(self.sketch_emb - normal_emb, dim=-1)
            self.log("optimize/siamese_loss", loss, on_step=True)

            # visualize sketch and the current normal image
            self.log_image("normal", self.deepsdf.siamese_input_to_image(normal))
            self.log_image("sketch", self.deepsdf.siamese_input_to_image(self.sketch))
