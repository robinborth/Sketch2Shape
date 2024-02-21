import torch

from lib.models.loss import Loss


class Siamese(Loss):
    def __init__(
        self,
        margin: float = 0.2,
        reg_loss: bool = True,
        reg_weight: float = 1e-03,
        support_latent: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    def model_step(self, batch, batch_idx: int, split: str = "train"):
        emb = self.forward(batch["image"], batch["type_idx"])  # (B, D)

        # calculate the anchor, positive, negative indx
        a_idx, p_idx, n_idx = self.get_all_triplets_indices(batch["label"])

        # calculate the triplet loss
        d_ap = torch.linalg.vector_norm(emb[a_idx] - emb[p_idx], dim=-1)
        self.log(f"{split}/distance_anchor_positive", d_ap.mean())
        d_an = torch.linalg.vector_norm(emb[a_idx] - emb[n_idx], dim=-1)
        self.log(f"{split}/distance_anchor_negative", d_an.mean())

        # calculate how many pairs would be classified wrong
        incorrect_count = ((d_ap - d_an) > 0).sum().float()
        self.log(f"{split}/incorrect_count", incorrect_count, prog_bar=True)

        m = self.hparams["margin"]
        triplet_loss = torch.relu(d_ap - d_an + m)  # max(0, d_ap - d_an + m)
        triplet_mask = triplet_loss > 0

        # triplet_loss = triplet_loss[triplet_mask].mean()  # no zero avg
        triplet_loss = triplet_loss.mean()  # full avg
        self.log(f"{split}/triplet_loss", triplet_loss)

        triplet_count = triplet_mask.sum().float()
        self.log(f"{split}/triplet_count", triplet_count)

        # calculate the reg loss based on the embeeddings
        reg_loss = torch.tensor(0).to(triplet_loss)
        if self.hparams["reg_loss"]:
            reg_loss = torch.linalg.vector_norm(emb, dim=-1).mean()
            reg_loss *= self.hparams["reg_weight"]
            self.log(f"{split}/reg_loss", reg_loss)

        # compute the final loss
        loss = reg_loss + triplet_loss
        self.log(f"{split}/loss", loss, prog_bar=True)

        return loss
