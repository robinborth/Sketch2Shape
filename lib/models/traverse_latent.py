from pathlib import Path

import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

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
        # siamese settings (optional)
        data_dir: str = "/data",
        siamese_ckpt_path: str = "siamese.ckpt",
        shape_k: int = 16,
        shape_view_id: int = 11,
        shape_init: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.meshes: list[dict] = []

        latent_start = None
        if Path(siamese_ckpt_path).exists():
            self.siamese = Siamese.load_from_checkpoint(siamese_ckpt_path)
            self.siamese.freeze()
            self.siamese.eval()
            device = self.siamese.device
            metainfo = MetaInfo(data_dir=data_dir)
            sketch = metainfo.load_image(prior_idx_start, shape_view_id, 0)
            transforms = SiameseTransform(mean=0.5, std=0.5)
            self.sketch = transforms(sketch)[None, ...].to(device)
            self.sketch_emb = self.siamese(self.sketch)
            if shape_init:
                metainfo = MetaInfo(data_dir=data_dir, split="train")
                _loss = []
                for obj_id in tqdm(metainfo.obj_ids):
                    label = int(metainfo.label_to_obj_id(obj_id))
                    normal = metainfo.load_image(label, shape_view_id, 1)  # normal
                    normal_emb = self.siamese(transforms(normal)[None, ...].to(device))
                    snn_loss = 1 - cosine_similarity(self.sketch_emb, normal_emb)
                    _loss.append(snn_loss)
                loss = torch.concatenate(_loss)
                idx = torch.argsort(loss)[:shape_k]
                shape_latents = self.deepsdf.lat_vecs.weight[idx].mean(0)
                latent_start = shape_latents.mean(0)

        if latent_start is None:
            latent_start = self.deepsdf.get_latent(prior_idx_start)
        self.register_buffer("latent_start", latent_start)

        latent_end = self.deepsdf.get_latent(prior_idx_end)
        self.register_buffer("latent_end", latent_end)

    def validation_step(self, batch, batch_idx):
        t = batch[0]  # t = [0, 1]
        self.latent = (1 - t) * self.latent_start + t * self.latent_end  # interpolate

        if self.hparams["create_mesh"]:
            mesh = self.to_mesh()
            self.meshes.append(mesh)

        if self.hparams["create_video"]:
            self.capture_camera_frame()

        if self.snn_loss:
            # calculate the normal embedding
            rendered_normal = self.capture_camera_frame()  # (H, W, 3)
            normal = self.deepsdf.normal_to_siamese(rendered_normal)  # (1, 3, H, W)
            normal_emb = self.siamese(normal)
            normal_norm = torch.norm(normal_emb, dim=-1)
            self.log("optimize/rendered_norm", normal_norm, on_step=True)

            # calculate the siamese loss
            loss = 1 - cosine_similarity(self.sketch_emb, normal_emb)
            self.log("optimize/siamese_loss", loss, on_step=True)

            # visualize sketch and the current normal image
            self.log_image("normal", self.deepsdf.siamese_input_to_image(normal))
            self.log_image("sketch", self.deepsdf.siamese_input_to_image(self.sketch))
