import torch
from torch.nn.functional import cosine_similarity

from lib.optimizer.latent import LatentOptimizer


class LatentTraversal(LatentOptimizer):
    def __init__(
        self,
        # init settings: random, mean, prior, prior(idx), retrieval, latent
        source_latent_init: str = "random",
        source_obj_id: str = "",
        # init settings: random, mean, prior, prior(idx), retrieval, latent
        target_latent_init: str = "random",
        target_obj_id: str = "",
        # video settings
        create_mesh: bool = True,
        create_video: bool = True,
        compute_loss: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.meshes: list[dict] = []

        # init the start latent code
        if source_latent_init.startswith("retrieval"):
            self.init_retrieval_latents(
                obj_id=source_obj_id,
                view_id=self.hparams["prior_view_id"],
                k=self.hparams["retrieval_k"],
            )
        self.init_latent(
            name="latent_start",
            latent_init=source_latent_init,
            obj_id=source_obj_id,
        )

        # init the end latent code
        if target_latent_init.startswith("retrieval"):
            self.init_retrieval_latents(
                obj_id=target_obj_id,
                view_id=self.hparams["prior_view_id"],
                k=self.hparams["retrieval_k"],
            )
        self.init_latent(
            name="latent_end",
            latent_init=target_latent_init,
            obj_id=target_obj_id,
        )

    def validation_step(self, batch, batch_idx):
        t = batch[0]  # t = [0, 1]
        self.latent = (1 - t) * self.latent_start + t * self.latent_end  # interpolate

        if self.hparams["create_mesh"]:
            mesh = self.to_mesh()
            self.meshes.append(mesh)

        if self.hparams["create_video"]:
            self.capture_camera_frame()

        if self.hparams["compute_loss"]:
            # calculate the normal embedding
            rendered_normal = self.capture_camera_frame()  # (H, W, 3)
            normal = self.deepsdf.image_to_siamese(rendered_normal)  # (1, 3, H, W)
            normal_emb = self.loss(normal)
            normal_norm = torch.norm(normal_emb, dim=-1)
            self.log("optimize/rendered_norm", normal_norm, on_step=True)

            # calculate the siamese loss
            loss = 1 - cosine_similarity(self.sketch_emb, normal_emb)
            self.log("optimize/siamese_loss", loss, on_step=True)

            # visualize sketch and the current normal image
            self.log_image("normal", self.deepsdf.loss_input_to_image(normal))
            self.log_image("sketch", self.deepsdf.loss_input_to_image(self.sketch))
