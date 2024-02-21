import torch
from torch.nn.functional import cosine_similarity

from lib.optimizer.latent import LatentOptimizer


class SketchOptimizer(LatentOptimizer):
    def __init__(
        self,
        loss_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

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
        if self.hparams["reg_loss"] != "none":
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
        self.log_image("normal", self.deepsdf.loss_input_to_image(normal))
        self.log_image("sketch", self.deepsdf.loss_input_to_image(sketch))

        return total_loss
