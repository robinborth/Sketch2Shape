import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

from lib.data.metainfo import MetaInfo
from lib.data.transforms import BaseTransform
from lib.optimizer.latent import LatentOptimizer
from lib.utils.checkpoint import load_model


class SketchOptimizer(LatentOptimizer):
    def __init__(
        self,
        data_dir: str = "/data",
        loss_ckpt_path: str = "siamese.ckpt",
        shape_k: int = 32,
        shape_view_id: int = 11,  # this should be the same as in the dataset
        shape_init: bool = False,
        loss_weight: float = 1.0,
        shape_prior: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.loss = load_model(loss_ckpt_path)
        self.loss.freeze()
        self.loss.eval()

        transforms = BaseTransform(mean=0.5, std=0.5)
        device = self.loss.device
        if shape_init or shape_prior:
            metainfo = MetaInfo(data_dir=data_dir)
            obj_id_label = int(metainfo.obj_id_to_label(self.hparams["obj_id"]))
            sketch = metainfo.load_image(obj_id_label, shape_view_id, 0)  # sketch
            sketch_emb = self.loss(transforms(sketch)[None, ...].to(device))

            metainfo = MetaInfo(data_dir=data_dir, split="train")
            _loss = []
            for obj_id in tqdm(metainfo.obj_ids):
                label = int(metainfo.obj_id_to_label(obj_id))
                normal = metainfo.load_image(label, shape_view_id, 1)  # normal
                normal_emb = self.loss(transforms(normal)[None, ...].to(device))
                snn_loss = 1 - cosine_similarity(sketch_emb, normal_emb)
                _loss.append(snn_loss)
            self.shape_loss = torch.concatenate(_loss)
            # don't include the latent code that we want to optimize for
            idx = torch.argsort(self.shape_loss)
            shape_idx = idx[idx != obj_id_label][:shape_k]
            shape_latents = self.deepsdf.lat_vecs.weight[shape_idx]
            # remove outliers
            # std = shape_latents.std(0)
            # mean = shape_latents.mean(0)
            # shape_lats_idx = ((shape_latents - mean) / std).pow(2).mean(-1) <= 1.0
            # shape_latents = shape_latents[shape_lats_idx]
            self.register_buffer("shape_latents", shape_latents)
            # self.shape_idx = shape_idx[shape_lats_idx]
            self.shape_idx = shape_idx

            if shape_init:
                latent = shape_latents.mean(0)
                self.register_buffer("latent", latent)

    def training_step(self, batch, batch_idx):
        self.loss.eval()

        # get the gt image and normals
        sketch = batch["sketch"]  # dim (1, 3, H, W) and values are (-1, 1)
        sketch_emb = self.loss(sketch)  # (1, D)

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
        normal_emb = self.loss(normal)  # (1, D)

        loss = 1 - cosine_similarity(sketch_emb, normal_emb).clone()
        loss *= self.hparams["loss_weight"]
        self.log("optimize/loss", loss)

        reg_loss = torch.tensor(0).to(loss)
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

        total_loss = reg_loss.mean() + loss.mean()
        self.log("optimize/total_loss", total_loss, prog_bar=True)

        latent_norm = torch.norm(self.latent, dim=-1)
        self.log("optimize/latent_norm", latent_norm)

        # visualize the different images
        self.log_image("normal", self.deepsdf.siamese_input_to_image(normal))
        self.log_image("sketch", self.deepsdf.siamese_input_to_image(sketch))

        return total_loss
