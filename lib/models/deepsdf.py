import math
from typing import Any, List

import lightning as L
import torch
import torch.nn as nn


class DeepSDF(L.LightningModule):
    def __init__(
        self,
        loss: torch.nn.Module,
        decoder_optimizer: torch.optim.Optimizer,
        latents_optimizer: torch.optim.Optimizer,
        latent_size: int = 512,
        num_hidden_layers: int = 8,
        latent_vector_size: int = 256,
        clamp: bool = True,
        clamp_val: float = 0.1,
        reg_loss: bool = True,
        num_scenes: int = 1,
        sigma: float = 1e-4,
        skip_connection: list[int] = [4],
        dropout: bool = True,
        dropout_p: float = 0.2,
        dropout_latent: bool = False,
        dropout_latent_p: float = 0.2,
        weight_norm: bool = False,
        decoder_scheduler=None,
        latents_scheduler=None,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.loss = loss

        self._build_model()

        # latent vectors
        self.lat_vecs = nn.Embedding(
            self.hparams["num_scenes"], self.hparams["latent_vector_size"]
        )
        std_lat_vec = 1.0 / math.sqrt(self.hparams["latent_vector_size"])
        torch.nn.init.normal_(self.lat_vecs.weight.data, 0.0, std_lat_vec)

        # Whether to use schedulers or not
        self.schedulers = decoder_scheduler is not None

        # latent dropout
        if self.hparams["dropout_latent"]:
            self.latent_dropout = nn.Dropout(p=self.hparams["dropout_latent_p"])

    def forward(self, x):
        """
        x is a tuple with the coordinates at position 0 and latent_vectors at position 1
        """
        if self.hparams["dropout_latent"]:
            x[1] = self.latent_dropout(x[1])
        out = torch.cat(x, dim=2)
        for i, layer in enumerate(self.decoder):
            if i in self.hparams["skip_connection"]:
                _skip = torch.cat(x, dim=2)
                out = torch.cat((out, _skip), dim=2)
                out = layer(out)
            else:
                out = layer(out)
        return out

    def training_step(self, batch, batch_idx):
        xyz = batch["xyz"]
        y = batch["sd"].flatten()
        lat_vec = self.lat_vecs(batch["idx"])

        opt1, opt2 = self.optimizers()
        opt1.zero_grad()
        opt2.zero_grad()

        y_hat = self.forward((xyz, lat_vec)).flatten()

        if self.hparams["clamp"]:
            y_hat = torch.clamp(
                y_hat, -self.hparams["clamp_val"], self.hparams["clamp_val"]
            )
            y = torch.clamp(y, -self.hparams["clamp_val"], self.hparams["clamp_val"])

        l1_loss = self.loss(y, y_hat)
        self.log("train/l1_loss", l1_loss, on_step=True, on_epoch=True)

        if self.hparams["reg_loss"]:
            reg_loss_sum = torch.sum(torch.linalg.norm(lat_vec))
            reg_loss = (
                reg_loss_sum
                * min(1, 1 / (self.current_epoch + 1))
                * self.hparams["sigma"]
            ) / y.shape[0]
            self.log("train/reg_loss", reg_loss, on_step=True, on_epoch=True)
            loss = l1_loss + reg_loss
        else:
            loss = l1_loss

        self.manual_backward(loss)
        opt1.step()
        opt2.step()
        if self.schedulers:
            sch1, sch2 = self.lr_schedulers()
            sch1.step()
            sch2.step()
            # state = self.trainer.optimizers[0].state
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def predict(self, x):
        out = torch.cat(x, dim=-1)
        for i, layer in enumerate(self.decoder):
            if i in self.hparams["skip_connection"]:
                _skip = torch.cat(x, dim=-1)
                out = torch.cat((out, _skip), dim=-1)
            out = layer(out)
        return out

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

    def _build_model(self):
        # build the mlp
        layers = list()
        layers.append(
            nn.Sequential(
                nn.utils.parametrizations.weight_norm(
                    nn.Linear(
                        3 + self.hparams["latent_vector_size"],
                        self.hparams["latent_size"],
                    )
                )
                if self.hparams["weight_norm"]
                else nn.Linear(
                    3 + self.hparams["latent_vector_size"], self.hparams["latent_size"]
                ),
                nn.ReLU(),
            )
        )
        for i in range(2, self.hparams["num_hidden_layers"]):
            if i in self.hparams["skip_connection"]:
                layers.append(
                    nn.Sequential(
                        nn.utils.parametrizations.weight_norm(
                            nn.Linear(
                                self.hparams["latent_size"],
                                self.hparams["latent_size"]
                                - self.hparams["latent_vector_size"]
                                - 3,
                            )
                        )
                        if self.hparams["weight_norm"]
                        else nn.Linear(
                            self.hparams["latent_size"],
                            self.hparams["latent_size"]
                            - self.hparams["latent_vector_size"]
                            - 3,
                        ),
                        nn.ReLU(),
                        nn.Dropout(p=self.hparams["dropout_p"]),
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        nn.utils.parametrizations.weight_norm(
                            nn.Linear(
                                self.hparams["latent_size"], self.hparams["latent_size"]
                            )
                        )
                        if self.hparams["weight_norm"]
                        else nn.Linear(
                            self.hparams["latent_size"], self.hparams["latent_size"]
                        ),
                        nn.ReLU(),
                        nn.Dropout(p=self.hparams["dropout_p"]),
                    )
                )
        layers.append(
            nn.Sequential(nn.Linear(self.hparams["latent_size"], 1), nn.Tanh())
        )
        self.decoder = nn.Sequential(*layers)
