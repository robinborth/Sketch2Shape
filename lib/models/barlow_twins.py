import torch

from lib.models.loss import Loss


class BarlowTwins(Loss):
    def __init__(
        self,
        gamma: float = 5e-03,
        support_latent: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    def model_step(self, batch, batch_idx: int, split: str = "train"):
        # compute embeddings
        N = batch["image"].shape[0] // 2
        emb = self.forward(batch["image"], batch["type_idx"])  # (B, D)

        # calculate the anchor, positive, negative indx
        a_idx, p_idx, n_idx = self.get_all_triplets_indices(batch["label"])

        # calculate the triplet loss
        d_ap = torch.norm(emb[a_idx] - emb[p_idx], dim=-1)
        self.log(f"{split}/distance_anchor_positive", d_ap.mean())
        d_an = torch.norm(emb[a_idx] - emb[n_idx], dim=-1)
        self.log(f"{split}/distance_anchor_negative", d_an.mean())

        # calculate how many pairs would be classified wrong
        incorrect_count = ((d_ap - d_an) > 0).sum().float()
        self.log(f"{split}/incorrect_count", incorrect_count, prog_bar=True)

        # normalize repr. along the batch dimension
        a_idx, b_idx = self.get_augmentations_idx(batch["label"])
        z_a_norm = (emb[a_idx] - emb[a_idx].mean(0)) / emb[a_idx].std(0)  # NxD
        z_b_norm = (emb[b_idx] - emb[b_idx].mean(0)) / emb[b_idx].std(0)  # NxD

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N  # DxD
        diag_mask = torch.eye(*c.shape, dtype=torch.bool)

        # loss
        invariance_loss = (1 - c[diag_mask]).pow(2).sum()
        self.log(f"{split}/invariance_loss", invariance_loss)

        redundancy_loss = (c[~diag_mask]).pow(2).sum() * self.hparams["gamma"]
        self.log(f"{split}/redundancy_loss", redundancy_loss)

        loss = (invariance_loss + redundancy_loss) / N
        self.log(f"{split}/loss", loss, prog_bar=True)

        return loss
