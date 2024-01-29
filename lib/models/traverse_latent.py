from lib.models.optimize_latent import LatentOptimizer


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
