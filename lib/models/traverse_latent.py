import torch

from lib.models.optimize_latent import LatentOptimizer


class DeepSDFLatentTraversal(LatentOptimizer):
    def __init__(
        self,
        prior_idx_start: int = -1,
        prior_idx_end: int = -1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.meshes: list[dict] = []

    def validation_step(self, batch, batch_idx):
        latent = self.latent.clone()
        t = batch[0]  # t = [0, 1]

        latent_start = self.latent  # mean latent
        if (idx_start := self.hparams["prior_idx_start"]) >= 0:
            idx_start = torch.tensor(idx_start).to(self.latent.device)
            latent_start = self.model.lat_vecs(idx_start)

        latent_end = self.latent  # mean latent
        if (idx_end := self.hparams["prior_idx_end"]) >= 0:
            idx_end = torch.tensor(idx_end).to(self.latent.device)
            latent_end = self.model.lat_vecs(idx_end)

        # override the latent for inference
        self.latent = t * latent_start + (1 - t) * latent_end
        mesh = self.to_mesh()
        self.meshes.append(mesh)

        # restore the mean latent
        self.latent = latent
