import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule


class DeepSDF(LightningModule):
    # TODO should we weight the negative examples more?
    def __init__(
        self,
        loss: torch.nn.Module,
        decoder_optimizer: torch.optim.Optimizer,
        latents_optimizer: torch.optim.Optimizer,
        decoder_scheduler=None,
        latents_scheduler=None,
        latent_size: int = 512,
        num_hidden_layers: int = 8,
        latent_vector_size: int = 256,
        num_latent_vectors: int = 1,
        clamp: bool = True,
        clamp_val: float = 0.1,
        reg_loss: bool = True,
        reg_weight: float = 1e-4,
        skip_connection: list[int] = [4],
        dropout: float = 0.0,
        weight_norm: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False

        self.loss = loss()
        self.schedulers = decoder_scheduler is not None

        # inital layers and first input layer
        layers = []  # type: ignore
        layer = nn.Linear(3 + latent_vector_size, latent_size)
        if weight_norm:
            layer = nn.utils.parametrizations.weight_norm(layer)
        layers.append(nn.Sequential(layer, nn.ReLU()))

        # backbone layers
        for layer_idx in range(2, num_hidden_layers + 1):
            output_size = latent_size
            if layer_idx in skip_connection:
                output_size = latent_size - latent_vector_size - 3
            layer = nn.Linear(latent_size, output_size)
            if weight_norm:
                layer = nn.utils.parametrizations.weight_norm(layer)
            layers.append(nn.Sequential(layer, nn.ReLU(), nn.Dropout(p=dropout)))

        # # output layer and final deepsdf backbone
        layers.append(nn.Sequential(nn.Linear(latent_size, 1), nn.Tanh()))
        self.decoder = nn.Sequential(*layers)

        # latent vectors
        self.lat_vecs = nn.Embedding(num_latent_vectors, latent_vector_size)
        std_lat_vec = 1.0 / np.sqrt(latent_vector_size)
        torch.nn.init.normal_(self.lat_vecs.weight.data, 0.0, std_lat_vec)

    def forward(self, points: torch.Tensor, latent: torch.Tensor, mask=None):
        """The forward pass of the deepsdf model.

        Args:
            points (torch.Tensor): The points of dim (B, N, 3) or (N, 3).
            latent (torch.Tensor): The latent code of dim (B, L) or (L).
            mask (torch.Tensor): The mask before feeding to model of dim or (N). Make
            sure that the dimension of the latent is (L) and of the points (N, 3).

        Returns:
            torch.Tensor: The sdf values of dim (B, N) or (N) or when mask applied the
            dimension depends on the positive entries of the mask hence dim ([0...N]).
        """
        N, L = points.shape[-2], latent.shape[-1]

        if len(latent.shape) == 1:
            latent = latent.unsqueeze(-2).expand(N, L)
        else:
            latent = latent.unsqueeze(-2).expand(-1, N, L)
            assert mask is None  # check that only without batching

        if mask is not None:
            points, latent = points[mask], latent[mask]
        out = torch.cat((points, latent), dim=-1)

        for layer_idx, layer in enumerate(self.decoder):
            if layer_idx in self.hparams["skip_connection"]:
                _skip = torch.cat((points, latent), dim=-1)
                out = torch.cat((out, _skip), dim=-1)
            out = layer(out)

        return out.squeeze(-1)

    def training_step(self, batch, batch_idx):
        opt1, opt2 = self.optimizers()
        opt1.zero_grad()
        opt2.zero_grad()

        gt_sdf = batch["sdf"]  # (B, N)
        points = batch["points"]  # (B, N, 3)
        latent = self.lat_vecs(batch["idx"])  # (B, L)

        sdf = self.forward(points=points, latent=latent)  # (B, N)

        if self.hparams["clamp"]:
            clamp_val = self.hparams["clamp_val"]
            sdf = torch.clamp(sdf, -clamp_val, clamp_val)
            gt_sdf = torch.clamp(gt_sdf, -clamp_val, clamp_val)

        l1_loss = self.loss(sdf, gt_sdf)
        self.log("train/l1_loss", l1_loss, on_step=True, on_epoch=True)

        reg_loss = torch.tensor(0).to(l1_loss)
        if self.hparams["reg_loss"]:
            reg_loss = torch.linalg.norm(latent, dim=-1).mean()
            reg_loss *= min(1, self.current_epoch / 100)  # TODO add to hparams
            reg_loss *= self.hparams["reg_weight"]
            self.log("train/reg_loss", reg_loss, on_step=True, on_epoch=True)

        loss = l1_loss + reg_loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        self.manual_backward(loss)
        opt1.step()
        opt2.step()

        return loss

    def on_train_epoch_end(self):
        if self.schedulers:
            sch1, sch2 = self.lr_schedulers()
            sch1.step()
            sch2.step()

    def configure_optimizers(self):
        optim_decoder = self.hparams["decoder_optimizer"](self.decoder.parameters())
        optim_latents = self.hparams["latents_optimizer"](self.lat_vecs.parameters())
        if self.schedulers:
            scheduler_decoder = self.hparams["decoder_scheduler"](optim_decoder)
            scheduler_latents = self.hparams["latents_scheduler"](optim_latents)
            return (
                {"optimizer": optim_decoder, "lr_scheduler": scheduler_decoder},
                {"optimizer": optim_latents, "lr_scheduler": scheduler_latents},
            )
        return [optim_decoder, optim_latents]
