import torch

from lib.models.optimize_latent import LatentOptimizer
from lib.models.siamese import Siamese


class DeepSDFSketchRender(LatentOptimizer):
    def __init__(
        self,
        siamese_ckpt_path: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.siamese = Siamese.load_from_checkpoint(siamese_ckpt_path)
        self.siamese.freeze()
        self.model.lat_vecs = None

    def training_step(self, batch, batch_idx):
        self.siamese.eval()

        # get the gt image and normals
        gt_image = batch["gt_image"].squeeze()
        # TODO what is this line?
        # gt_normals = self.image_to_normal(gt_image)
        unit_sphere_mask = batch["mask"].squeeze()
        camera_pos = batch["camera_pos"].squeeze()
        lightsource = batch["lightsource"]
        obj_reflection = batch["obj_reflection"]

        # calculate the normals map
        points, surface_mask = self.sphere_tracing(
            points=batch["points"].squeeze(),
            rays=batch["rays"].squeeze(),
            mask=unit_sphere_mask,
        )
        normals = self.render_object(
            points=points,
            mask=surface_mask,
            camera_pos=camera_pos,
            lightsource=lightsource,
            obj_reflection=obj_reflection,
        )
        # TODO rename normal_to_image
        image = self.normal_to_image(normals, surface_mask)
        # calculate the loss for the object and usefull information to wandb

        gt_emb = self.siamese.decoder(gt_image.reshape(-1, 3, 256, 256))
        image_emb = self.siamese.decoder(image.reshape(-1, 3, 256, 256))
        image_loss = -torch.nn.functional.cosine_similarity(gt_emb, image_emb, dim=-1)
        self.log("optimize/image_loss", image_loss)

        # visualize the latent norm
        latent_norm = torch.linalg.norm(self.latent, dim=-1)
        self.log("optimize/latent_norm", latent_norm)

        # add regularization loss
        reg_loss = torch.tensor(0).to(image_loss)
        if self.hparams["reg_loss"]:
            reg_loss = latent_norm * self.hparams["reg_weight"]
            self.log("optimize/reg_loss", reg_loss)

        # log the full loss
        loss = reg_loss + image_loss
        self.log("optimize/loss", loss)

        # visualize the different images
        self.log_image("gt_image", gt_image)
        self.log_image("image", image)

        return loss
