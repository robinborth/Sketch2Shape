import torch

from lib.optimizer.latent import LatentOptimizer


class SketchOptimizer(LatentOptimizer):
    def __init__(
        self,
        loss_weight: float = 1.0,
        silhouette_loss: bool = True,
        silhouette_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    def training_step(self, batch, batch_idx):
        self.loss.eval()

        # get the gt image and normals
        sketch = batch["sketch"]  # dim (1, 3, H, W) and values are (-1, 1)
        sketch_emb = self.loss.embedding(sketch, mode="sketch")  # (1, D)

        # calculate the points on the surface
        with torch.no_grad():
            points, surface_mask = self.deepsdf.sphere_tracing(
                latent=self.latent,
                points=batch["points"].squeeze(),
                rays=batch["rays"].squeeze(),
                mask=batch["mask"].squeeze(),
            )

        # render the normals image
        rendered_normals = self.deepsdf.render_normals(
            latent=self.latent,
            points=points,
            mask=surface_mask,
        )  # (H, W, 3)

        # render the grayscale image and get the embedding from the grayscale image
        rendered_grayscale = self.deepsdf.normal_to_grayscale(
            normal=rendered_normals,
            ambmient=self.deepsdf.hparams["ambient"],
            diffuse=self.deepsdf.hparams["diffuse"],
        )  # (H, W, 3)
        grayscale = self.deepsdf.image_to_siamese(rendered_grayscale)  # (1, 3, H, W)
        grayscale_emb = self.loss.embedding(grayscale, mode="grayscale")  # (1, D)

        # calculate the loss between the sketch and the grayscale image
        loss = self.loss.compute(sketch_emb, grayscale_emb).clone()
        loss *= self.hparams["loss_weight"]
        self.log("optimize/loss", loss)

        # apply the regularization loss
        reg_loss = torch.tensor(0.0).to(loss)
        if self.hparams["reg_loss"] != "none":
            if self.hparams["reg_loss"] == "retrieval":
                std = self.shape_latents.std(0)
                mean = self.shape_latents.mean(0)
                reg_loss = ((self.latent.clone() - mean) / std).pow(2)
            elif self.hparams["reg_loss"] == "prior":
                std = self.deepsdf.lat_vecs.weight.std(0)
                mean = self.deepsdf.lat_vecs.weight.mean(0)
                reg_loss = ((self.latent.clone() - mean) / std).pow(2)
            elif self.hparams["reg_loss"] == "latent":
                reg_loss = torch.nn.functional.l1_loss(sketch_emb, self.latent)
            self.log("optimize/reg_loss_abs_max", torch.abs(reg_loss).max())
            reg_loss *= self.hparams["reg_weight"]
            self.log("optimize/reg_loss", reg_loss.mean())

        # calculate the silhouette loss to fix missing parts and grow in empty space
        silhouette_loss = torch.tensor(0.0).to(loss)
        if self.hparams["silhouette_loss"]:
            out = self.deepsdf.render_silhouette(
                normals=rendered_normals,
                points=points,
                latent=self.latent,
                return_full=True,
            )
            soft_silhouette = out["min_sdf"] - self.deepsdf.hparams["surface_eps"]
            silhouette_loss = out["weighted_silhouette"] * torch.relu(soft_silhouette)
            silhouette_loss *= self.hparams["silhouette_weight"]
        # TODO

        total_loss = reg_loss.mean() + loss.mean()
        self.log("optimize/total_loss", total_loss, prog_bar=True)

        latent_norm = torch.norm(self.latent, dim=-1)
        self.log("optimize/latent_norm", latent_norm)

        # visualize the different images
        self.log_image("grayscale", self.deepsdf.loss_input_to_image(grayscale))
        self.log_image("sketch", self.deepsdf.loss_input_to_image(sketch))

        return total_loss
