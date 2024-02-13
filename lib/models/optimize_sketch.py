import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

from lib.data.metainfo import MetaInfo
from lib.data.transforms import SiameseTransform
from lib.models.optimize_latent import LatentOptimizer
from lib.models.siamese import Siamese


class DeepSDFSketchRender(LatentOptimizer):
    def __init__(
        self,
        data_dir: str = "/data",
        siamese_ckpt_path: str = "siamese.ckpt",
        shape_k: int = 16,
        shape_view_id: int = 11,  # this should be the same as in the dataset
        shape_init: bool = False,
        siamese_weight: float = 1.0,
        shape_prior: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.siamese = Siamese.load_from_checkpoint(siamese_ckpt_path)
        self.siamese.freeze()
        self.siamese.eval()

        transforms = SiameseTransform(mean=0.5, std=0.5)
        device = self.siamese.device
        if shape_init or shape_prior:
            metainfo = MetaInfo(data_dir=data_dir)
            obj_id_label = int(metainfo.obj_id_to_label(self.hparams["obj_id"]))
            sketch = metainfo.load_image(obj_id_label, shape_view_id, 0)  # sketch
            sketch_emb = self.siamese(transforms(sketch)[None, ...].to(device))

            metainfo = MetaInfo(data_dir=data_dir, split="train")
            _loss = []
            for obj_id in tqdm(metainfo.obj_ids):
                label = int(metainfo.obj_id_to_label(obj_id))
                normal = metainfo.load_image(label, shape_view_id, 1)  # normal
                normal_emb = self.siamese(transforms(normal)[None, ...].to(device))
                snn_loss = 1 - cosine_similarity(sketch_emb, normal_emb)
                _loss.append(snn_loss)
            self.shape_loss = torch.concatenate(_loss)
            # don't include the latent code that we want to optimize for
            idx = torch.argsort(self.shape_loss)
            self.shape_idx = idx[idx != obj_id_label][:shape_k]

            shape_latents = self.deepsdf.lat_vecs.weight[self.shape_idx]
            self.register_buffer("shape_latents", shape_latents)

            if shape_init:
                latent = shape_latents.mean(0)
                self.register_buffer("latent", latent)

    def training_step(self, batch, batch_idx):
        self.siamese.eval()

        # get the gt image and normals
        sketch = batch["sketch"]  # dim (1, 3, H, W) and values are (-1, 1)
        sketch_emb = self.siamese(sketch)  # (1, D)

        # calculate the normals map and embedding
        points, surface_mask = self.deepsdf.sphere_tracing(
            latent=self.latent,
            points=batch["points"].squeeze(),
            rays=batch["rays"].squeeze(),
            mask=batch["mask"].squeeze(),
        )
        rendered_normal = self.deepsdf.render_normals(
            latent=self.latent,
            points=points,
            mask=surface_mask,
        )  # (H, W, 3)
        normal = self.deepsdf.normal_to_siamese(rendered_normal)  # (1, 3, H, W)
        normal_emb = self.siamese(normal)  # (1, D)

        siamese_loss = 1 - cosine_similarity(sketch_emb, normal_emb).clone()
        siamese_loss *= self.hparams["siamese_weight"]
        self.log("optimize/siamese_loss", siamese_loss)

        reg_loss = torch.tensor(0).to(siamese_loss)
        if self.hparams["reg_loss"]:
            if self.hparams["shape_prior"]:
                std = self.shape_latents.std(0)
                mean = self.shape_latents.mean(0)
                reg_loss = ((self.latent.clone() - mean) / std).pow(2)
            else:
                std = self.deepsdf.lat_vecs.weight.std(0)
                mean = self.deepsdf.lat_vecs.weight.mean(0)
                reg_loss = ((self.latent.clone() - mean) / std).pow(2)
            self.log("optimize/reg_loss_abs_max", torch.abs(reg_loss).max())

            # reg_loss = torch.norm(self.latent, dim=-1).clone()
            reg_loss *= self.hparams["reg_weight"]
            self.log("optimize/reg_loss", reg_loss.mean())

        loss = reg_loss.mean() + siamese_loss.mean()
        self.log("optimize/loss", loss, prog_bar=True)

        latent_norm = torch.norm(self.latent, dim=-1)
        self.log("optimize/latent_norm", latent_norm)

        # visualize the different images
        self.log_image("normal", self.deepsdf.siamese_input_to_image(normal))
        self.log_image("sketch", self.deepsdf.siamese_input_to_image(sketch))

        return loss
