import torch

from lib.models.optimize_latent import LatentOptimizer


class DeepSDFLatentOptimizer(LatentOptimizer):
    def __init__(
        self,
        decoder_lr: float = 5e-04,
        latents_lr: float = 1e-03,
        reg_loss: bool = True,
        reg_weight: float = 1e-4,
        clamp: bool = True,
        clamp_val: float = 0.1,
        adaptive_sample_strategy: bool = False,
        adaptive_mining_strategy: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.deepsdf.lat_vecs = None

    def training_step(self, batch, batch_idx):
        gt_sdf = batch["sdf"].squeeze()  # (N)
        points = batch["points"].squeeze()  # (N, 3)

        sdf = self.forward(points=points)  # (N)
        if self.hparams["clamp"]:
            clamp_val = self.hparams["clamp_val"]
            sdf = torch.clamp(sdf, -clamp_val, clamp_val)
            gt_sdf = torch.clamp(gt_sdf, -clamp_val, clamp_val)

        if self.hparams["adaptive_sample_strategy"]:  # curriculum deepsdf
            # values from curriculum deepsdf paper
            eps = 0.025
            if self.current_epoch > 200:
                eps = 0.01
            if self.current_epoch > 600:
                eps = 0.0025
            if self.current_epoch > 1000:
                eps = 0.0
            zero = torch.tensor(0).to(self.device)
            loss = torch.max(torch.abs(gt_sdf - sdf) - eps, zero)
            self.log("train/l1_loss", loss.mean(), on_step=True)
            drop_ratio = (loss == 0.0).sum() / loss.numel()
            self.log("train/drop_ratio", drop_ratio, on_step=True)
        else:  # default deepsdf loss
            loss = torch.nn.functional.l1_loss(sdf, gt_sdf, reduction="none")
            self.log("train/l1_loss", loss.mean(), on_step=True)

        if self.hparams["adaptive_mining_strategy"]:
            # values from curriculum deepsdf paper
            gamma = 0
            if self.current_epoch > 200:
                gamma = 0.1
            if self.current_epoch > 600:
                gamma = 0.2
            if self.current_epoch > 1000:
                gamma = 0.5
            mining_weight = 1 + (gamma * torch.sign(gt_sdf) * torch.sign(gt_sdf - sdf))
            self.log("train/mining_weight", mining_weight.mean(), on_step=True)
            loss *= mining_weight

        reg_loss = torch.tensor(0).to(loss)
        if self.hparams["reg_loss"]:
            reg_loss = torch.norm(self.latent, dim=-1).clone()
            reg_loss *= min(1, self.current_epoch / 100)
            reg_loss *= self.hparams["reg_weight"]
            self.log("train/reg_loss", reg_loss.mean(), on_step=True)

        final_loss = loss.mean() + reg_loss.mean()
        self.log("train/loss", final_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_wo_reg_loss", loss.mean(), on_step=True)

        return final_loss
