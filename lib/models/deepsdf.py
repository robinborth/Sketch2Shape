import math
from typing import Any, List

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import trimesh
from skimage.measure import marching_cubes


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

        l1_loss = self.loss(y_hat, y)
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
        # state = self.trainer.optimizers[0].state

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def on_train_epoch_end(self):
        if self.schedulers:
            sch1, sch2 = self.lr_schedulers()
            sch1.step()
            sch2.step()

    def predict(self, x):
        with torch.no_grad():
            out = torch.cat(x, dim=2)
            for i, layer in enumerate(self.decoder):
                if i in self.hparams["skip_connection"]:
                    _skip = torch.cat(x, dim=2)
                    out = torch.cat((out, _skip), dim=2)
                    out = layer(out)
                else:
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


# reconstruction unknown shapes based on sdf values
class SDFReconstructor(L.LightningModule):
    def __init__(
        self,
        ckpt_path: str,
        optim: torch.optim.Optimizer,
        loss: torch.nn.Module,
        scheduler=None,
        clamp: bool = True,
        clamp_val: float = 0.1,
        reg_loss: bool = True,
        sigma: float = 1e-4,
        prior_idx: int = -1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self._load_model()
        self._init_latent()

        self.loss = loss

    def _load_model(self):
        # TODO we only require the decoder weights on cuda, rest is irrelevant (esp. the lat_vecs)
        self.model = DeepSDF.load_from_checkpoint(
            self.hparams["ckpt_path"], map_location="cpu"
        )

    def _init_latent(self):
        if self.hparams["prior_idx"] >= 0:
            # using a prior
            self.latent = (
                self.model.lat_vecs(torch.tensor([self.hparams["prior_idx"]]))
                .detach()
                .cuda()
            )
            self.latent.requires_grad_()
        else:
            mean = self.model.lat_vecs.weight.mean(0)
            std = self.model.lat_vecs.weight.std(0)
            self.latent = torch.normal(mean.detach(), std.detach()).cuda()
            self.latent.requires_grad_()
        # save some GPU space
        self.model.lat_vecs = None

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        lat_vec = self.latent.expand(x.shape[1], -1).unsqueeze(0)
        return self.model.predict((x, lat_vec))

    def training_step(self, batch, batch_idx):
        self.model.eval()
        xyz = batch["xyz"]
        y = batch["sd"].flatten()
        lat_vec = self.latent.expand(y.shape[0], -1).unsqueeze(0)

        y_hat = self.forward((xyz, lat_vec)).squeeze()

        if self.hparams["clamp"]:
            y_hat = torch.clamp(
                y_hat, -self.hparams["clamp_val"], self.hparams["clamp_val"]
            )
            y = torch.clamp(y, -self.hparams["clamp_val"], self.hparams["clamp_val"])

        l1_loss = self.loss(y_hat, y)
        self.log("opt/l1_loss", l1_loss, on_step=True, on_epoch=True)
        if self.hparams["reg_loss"]:
            loss = l1_loss + self.hparams["sigma"] * torch.mean(self.latent.pow(2))
        else:
            loss = l1_loss
        self.log(
            "opt/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1
        )
        return loss

    def configure_optimizers(self):
        optim = self.hparams["optim"]([self.latent])
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optim)
            self.save_lr = scheduler.get_lr()
            return {"optimizer": optim, "lr_scheduler": scheduler}
        return optim

    def get_obj(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        self.latent.to(device)
        resolution = 256
        chunck_size = 500_000
        # hparams
        grid_vals = torch.arange(-1, 1, float(2 / resolution))
        grid = torch.meshgrid(grid_vals, grid_vals, grid_vals)

        xyz = torch.stack(
            (grid[0].ravel(), grid[1].ravel(), grid[2].ravel())
        ).transpose(1, 0)

        del grid, grid_vals

        # based on rough trial and error
        n_chunks = (xyz.shape[0] // chunck_size) + 1
        #    a.element_size() * a.nelement()

        xyz_chunks = xyz.unsqueeze(0).chunk(n_chunks, dim=1)
        sd_list = list()
        for _xyz in xyz_chunks:
            _xyz = _xyz.to(device)
            sd = self.predict(_xyz).squeeze().cpu().numpy()
            sd_list.append(sd)
        sd = np.concatenate(sd_list)
        sd_r = sd.reshape(resolution, resolution, resolution)

        verts, faces, _, _ = marching_cubes(sd_r, level=0.0)

        x_max = np.array([1, 1, 1])
        x_min = np.array([-1, -1, -1])
        verts = verts * ((x_max - x_min) / (resolution)) + x_min

        # Create a trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        # remove objects outside unit sphere
        # mesh = remove_faces_outside_sphere(mesh)

        path_obj = f"/home/korth/sketch2shape/temp/test2_epochs-{self.trainer.max_epochs}_lr-{self.save_lr}.obj"
        # Save the mesh as an OBJ file
        mesh.export(path_obj)
