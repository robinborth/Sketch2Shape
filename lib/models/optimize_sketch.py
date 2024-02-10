import torch

from lib.models.optimize_latent import LatentOptimizer
from lib.models.siamese import Siamese


class DeepSDFSketchRender(LatentOptimizer):
    def __init__(
        self,
        siamese_ckpt_path: str,
        siamese_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.siamese = Siamese.load_from_checkpoint(siamese_ckpt_path)
        self.siamese.freeze()
        self.siamese.eval()

        self.model.lat_vecs = None

    def training_step(self, batch, batch_idx):
        self.siamese.eval()

        # get the gt image and normals
        sketch = batch["sketch"]  # dim (1, 3, H, W) and values are (-1, 1)
        sketch_emb = self.siamese(sketch)  # (1, D)

        # calculate the normals map and embedding
        points, surface_mask = self.sphere_tracing(
            points=batch["points"].squeeze(),
            rays=batch["rays"].squeeze(),
            mask=batch["mask"].squeeze(),
        )
        rendered_normal = self.render_normals(
            points=points,
            mask=surface_mask,
        )  # (H, W, 3)
        normal = self.model.normal_to_siamese(rendered_normal)  # (1, 3, H, W)
        normal_emb = self.siamese(normal)  # (1, D)

        siamese_loss = torch.norm(sketch_emb - normal_emb, dim=-1).clone()
        siamese_loss *= self.hparams["siamese_weight"]
        self.log("optimize/siamese_loss", siamese_loss)

        reg_loss = torch.tensor(0).to(siamese_loss)
        if self.hparams["reg_loss"]:
            reg_loss = torch.norm(self.latent, dim=-1).clone()
            reg_loss *= self.hparams["reg_weight"]
            self.log("optimize/reg_loss", reg_loss)

        loss = reg_loss + siamese_loss
        self.log("optimize/loss", loss)

        # visualize the different images
        self.log_image("normal", self.model.siamese_input_to_image(normal))
        self.log_image("sketch", self.model.siamese_input_to_image(sketch))

        return loss
