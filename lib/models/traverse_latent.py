import torch
from torch.nn.functional import l1_loss

from lib.data.metainfo import MetaInfo
from lib.data.transforms import SketchTransform
from lib.models.optimize_latent import LatentOptimizer
from lib.models.siamese import Siamese


class DeepSDFLatentTraversal(LatentOptimizer):
    def __init__(
        self,
        prior_idx_start: int = -1,
        prior_idx_end: int = -1,
        create_mesh: bool = True,
        create_video: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.meshes: list[dict] = []

        latent_start = self.model.get_latent(prior_idx_start)
        self.register_buffer("latent_start", latent_start)

        latent_end = self.model.get_latent(prior_idx_end)
        self.register_buffer("latent_end", latent_end)

        self.model.lat_vecs = None

    def validation_step(self, batch, batch_idx):
        t = batch[0]  # t = [0, 1]
        self.latent = (1 - t) * self.latent_start + t * self.latent_end  # interpolate

        if self.hparams["create_mesh"]:
            mesh = self.to_mesh()
            self.meshes.append(mesh)

        if self.hparams["create_video"]:
            self.capture_video_frame()


class DeepSDFSNNLatentTraversal(LatentOptimizer):
    def __init__(
        self,
        data_dir: str,
        siamese_ckpt_path: str,
        std: float = 1e-01,
        prior_idx_start: int = -1,
        prior_idx_end: int = -1,
        # prior_idx: int = -1,
        create_video: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.metainfo = MetaInfo(data_dir)
        transform = SketchTransform()
        sketch = self.metainfo.load_image(prior_idx_end, 11, 0)
        # sketch = transform(sketch)[None, ...]
        self.register_buffer("sketch", sketch)

        self.siamese = Siamese.load_from_checkpoint(siamese_ckpt_path)
        self.siamese.freeze()

        self.meshes: list[dict] = []

        latent_start = self.model.get_latent(prior_idx_start)
        self.register_buffer("latent_start", latent_start)

        latent_end = self.model.get_latent(prior_idx_end)
        self.register_buffer("latent_end", latent_end)

        # latent_start = self.model.get_latent(prior_idx)
        # noise = torch.rand_like(latent_start) * std
        # latent_start = latent_start + noise
        # self.register_buffer("latent_start", latent_start)

        # latent_end = self.model.get_latent(prior_idx)
        # self.register_buffer("latent_end", latent_end)

        # self.model.lat_vecs = None

    def validation_step(self, batch, batch_idx):
        self.siamese.eval()

        t = batch[0]  # t = [0, 1]
        self.latent = (1 - t) * self.latent_start + t * self.latent_end  # interpolate

        sketch = self.normal_to_image(self.sketch)
        self.log_image("sketch", sketch)

        sketch_emb = self.siamese(self.sketch)
        sketch_norm = torch.norm(sketch_emb, dim=-1)
        self.log("optimize/sketch_norm", sketch_norm, on_step=True)

        rendered_normals = self.capture_video_frame()
        normals = self.to_image(rendered_normals).permute(2, 0, 1)[None, ...]
        normal_emb = self.siamese(normals)
        normal_norm = torch.norm(normal_emb, dim=-1)
        self.log("optimize/normal_norm", normal_norm, on_step=True)

        loss = torch.norm(sketch_emb - normal_emb, dim=-1)
        self.log("optimize/siamese_loss", loss, on_step=True)
