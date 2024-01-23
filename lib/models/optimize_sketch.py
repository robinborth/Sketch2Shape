import torch
from torch.nn.functional import l1_loss

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
        self.model.lat_vecs = None

    def training_step(self, batch, batch_idx):
        self.siamese.eval()

        # get the gt image and normals
        sketch_image = batch["sketch"].squeeze(0)
        sketch = self.image_to_normal(sketch_image)
        unit_sphere_mask = batch["mask"].squeeze()

        # calculate the normals map
        points, surface_mask = self.sphere_tracing(
            points=batch["points"].squeeze(),
            rays=batch["rays"].squeeze(),
            mask=unit_sphere_mask,
        )
        rendered_normals = self.render_normals(points=points, mask=surface_mask)
        rendered_normals_image = self.normal_to_image(rendered_normals, surface_mask)

        # calculate the embeddings
        sketch_input = sketch.reshape(-1, 3, 256, 256)
        sketch_emb = self.siamese.decoder(sketch_input)
        rendered_normals_input = rendered_normals.reshape(-1, 3, 256, 256)
        rendered_normals_emb = self.siamese.decoder(rendered_normals_input)

        siamese_loss = l1_loss(sketch_emb, rendered_normals_emb)
        siamese_loss *= self.hparams["siamese_weight"]
        self.log("optimize/siamese_loss", siamese_loss)

        latent_norm = torch.linalg.norm(self.latent, dim=-1)
        self.log("optimize/latent_norm", latent_norm)

        reg_loss = torch.tensor(0).to(siamese_loss)
        if self.hparams["reg_loss"]:
            reg_loss = latent_norm * self.hparams["reg_weight"]
            self.log("optimize/reg_loss", reg_loss)

        loss = reg_loss + siamese_loss
        self.log("optimize/loss", loss)

        mem_allocated = torch.cuda.memory_allocated() / 1024**2  # convert to MIB
        self.log("optimize/mem_allocated", mem_allocated)

        # visualize the different images
        self.log_image("sketch_image", sketch_image)
        self.log_image("rendered_normals_image", rendered_normals_image)

        return loss
