from typing import Any, List

import lightning as L
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig


class DeepSDF(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # create the loss
        self.loss = instantiate(cfg.loss)

        # build the mlp
        layers: List[Any] = []
        layers.append(nn.Linear(3, cfg.latent_size))
        layers.append(nn.ReLU())
        for _ in range(cfg.num_hidden_layers):
            layers.append(nn.Linear(cfg.latent_size, cfg.latent_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(cfg.latent_size, 1))
        layers.append(nn.Tanh())
        self.backbone = nn.Sequential(*layers)

        # latent vectors
        # self.lat_vecs = nn.Embedding(num_scenes, self.cfg.latent_vector_size)
        # torch.nn.init.normal_(self.lat_vecs.weight.data, 0.0, 0.01)

    def forward(self, batch):
        out = {}
        out["sd"] = self.backbone(batch["xyz"]).flatten()
        return out

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.loss(output["sd"], batch["sd"])
        # state = self.trainer.optimizers[0].state
        self.log("train/loss", loss, prog_bar=True, batch_size=self.cfg.batch_size)
        return loss

    def predict(self, x):
        return self.backbone(x.to(torch.float32))

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            params=self.parameters(),
            lr=self.cfg.learning_rate,
        )
        # return optim
        # scheduler = instantiate(self.cfg.scheduler, optimizer=optim)
        return {
            "optimizer": optim
            # "lr_scheduler": scheduler,
            # "monitor": self.cfg.monitor,
        }


class ActualDeepSDF(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.automatic_optimization = False

        # create the loss
        self.loss = instantiate(cfg.loss)

        # build the mlp
        layers: List[Any] = []
        layers.append(
            nn.Sequential(
                nn.Linear(3 + cfg.latent_vector_size, cfg.latent_size), nn.ReLU()
            )
        )
        for i in range(2, cfg.num_hidden_layers):
            if i in self.cfg.skip_connection:
                layers.append(
                    nn.Sequential(
                        nn.Linear(
                            cfg.latent_size,
                            cfg.latent_size - cfg.latent_vector_size - 3,
                        ),
                        nn.ReLU(),
                        nn.Dropout(p=cfg.dropout_p),
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        nn.Linear(cfg.latent_size, cfg.latent_size),
                        nn.ReLU(),
                        nn.Dropout(p=cfg.dropout_p),
                    )
                )
        layers.append(nn.Sequential(nn.Linear(cfg.latent_size, 1), nn.Tanh()))
        self.decoder = nn.Sequential(*layers)

        # latent vectors
        self.lat_vecs = nn.Embedding(self.cfg.num_scenes, self.cfg.latent_vector_size)
        torch.nn.init.normal_(self.lat_vecs.weight.data, 0.0, 0.01)

    def forward(self, x):
        out = torch.cat(x, dim=2)
        for i, layer in enumerate(self.decoder):
            if i in self.cfg.skip_connection:
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

        if self.cfg.clamp:
            y_hat = torch.clamp(y_hat, -self.cfg.clamp_val, self.cfg.clamp_val)
            y = torch.clamp(y, -self.cfg.clamp_val, self.cfg.clamp_val)

        l1_loss = self.loss(y, y_hat)
        self.log("train/l1_loss", l1_loss)

        if self.cfg.reg_loss:
            reg_loss = torch.mean(torch.linalg.norm(lat_vec)) * self.cfg.sigma
            reg_loss /= y.shape[0]
            self.log("train/reg_loss", reg_loss)
            loss = l1_loss + reg_loss
        else:
            loss = l1_loss

        self.manual_backward(loss)
        opt1.step()
        opt2.step()
        # state = self.trainer.optimizers[0].state
        self.log("train/loss", loss, prog_bar=True, batch_size=self.cfg.batch_size)
        return loss

    def predict(self, x):
        return self.backbone(x)

    def configure_optimizers(self):
        optim_decoder = torch.optim.Adam(
            params=self.decoder.parameters(),
            lr=self.cfg.lr_decoder,
        )
        optim_latents = torch.optim.Adam(
            params=self.lat_vecs.parameters(), lr=self.cfg.lr_latents
        )
        # scheduler_decoder =
        # return optim
        # scheduler = instantiate(self.cfg.scheduler, optimizer=optim)
        return [optim_decoder, optim_latents]


### original imprlementation:
# # %%
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class Decoder(nn.Module):
#     def __init__(
#         self,
#         latent_size,
#         dims,
#         dropout=None,
#         dropout_prob=0.0,
#         norm_layers=[0,1,2,3,4,5,6,7],
#         latent_in=[4],
#         weight_norm=True,
#         xyz_in_all=False,
#         use_tanh=False,
#         latent_dropout=False,
#     ):
#         super(Decoder, self).__init__()

#         def make_sequence():
#             return []

#         dims = [latent_size + 3] + dims + [1]

#         self.num_layers = len(dims)
#         self.norm_layers = norm_layers
#         self.latent_in = latent_in
#         self.latent_dropout = latent_dropout
#         if self.latent_dropout:
#             self.lat_dp = nn.Dropout(0.2)

#         self.xyz_in_all = xyz_in_all
#         self.weight_norm = weight_norm

#         for layer in range(0, self.num_layers - 1):
#             if layer + 1 in latent_in:
#                 out_dim = dims[layer + 1] - dims[0]
#             else:
#                 out_dim = dims[layer + 1]
#                 if self.xyz_in_all and layer != self.num_layers - 2:
#                     out_dim -= 3

#             if weight_norm and layer in self.norm_layers:
#                 setattr(
#                     self,
#                     "lin" + str(layer),
#                     nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
#                 )
#             else:
#                 setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

#             if (
#                 (not weight_norm)
#                 and self.norm_layers is not None
#                 and layer in self.norm_layers
#             ):
#                 setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

#         self.use_tanh = use_tanh
#         if use_tanh:
#             self.tanh = nn.Tanh()
#         self.relu = nn.ReLU()

#         self.dropout_prob = dropout_prob
#         self.dropout = dropout
#         self.th = nn.Tanh()

#     # input: N x (L+3)
#     def forward(self, input):
#         xyz = input[:, -3:]

#         if input.shape[1] > 3 and self.latent_dropout:
#             latent_vecs = input[:, :-3]
#             latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
#             x = torch.cat([latent_vecs, xyz], 1)
#         else:
#             x = input

#         for layer in range(0, self.num_layers - 1):
#             lin = getattr(self, "lin" + str(layer))
#             if layer in self.latent_in:
#                 x = torch.cat([x, input], 1)
#             elif layer != 0 and self.xyz_in_all:
#                 x = torch.cat([x, xyz], 1)
#             x = lin(x)
#             # last layer Tanh
#             if layer == self.num_layers - 2 and self.use_tanh:
#                 x = self.tanh(x)
#             if layer < self.num_layers - 2:
#                 if (
#                     self.norm_layers is not None
#                     and layer in self.norm_layers
#                     and not self.weight_norm
#                 ):
#                     bn = getattr(self, "bn" + str(layer))
#                     x = bn(x)
#                 x = self.relu(x)
#                 if self.dropout is not None and layer in self.dropout:
#                     x = F.dropout(x, p=self.dropout_prob, training=self.training)

#         if hasattr(self, "th"):
#             x = self.th(x)

#         return x
