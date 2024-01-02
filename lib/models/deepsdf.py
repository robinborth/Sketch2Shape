import math
import os
from typing import Any, List

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import trimesh
from skimage.measure import marching_cubes

from lib.evaluate import compute_chamfer_distance
from lib.render.renderer import Renderer, normalize


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
            reg_loss = torch.mean(torch.linalg.norm(lat_vec, dim=2))
            reg_loss = (
                reg_loss * min(1, self.current_epoch / 100) * self.hparams["sigma"]
            )
            self.log("train/reg_loss", reg_loss, on_step=True, on_epoch=True)
            loss = l1_loss + reg_loss
        else:
            loss = l1_loss

        self.manual_backward(loss)
        opt1.step()
        opt2.step()

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
        for i in range(2, self.hparams["num_hidden_layers"] + 1):
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


# custom validation loop
class DeepSDFValidator(L.LightningModule):
    # TODO optimize the decoder performance
    # [ ] identify bottleneck using lightning simple profiler
    # [ ] https://lightning.ai/docs/pytorch/stable/tuning/profiler_intermediate.html
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
        resolution: int = 256,
        emp_init: bool = False,
        save_obj: bool = False,
        save_obj_path: str = "",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["loss"])

        self._load_model()
        # self._init_latent()

        self.loss = loss

        self.resolution = resolution
        self.idx2chamfer = dict()

        if self.hparams["save_obj"]:
            os.mkdir(self.hparams["save_obj_path"])

    def _load_model(self):
        self.model = DeepSDF.load_from_checkpoint(
            self.hparams["ckpt_path"], map_location="cpu"
        )
        self.model.freeze()

    def setup(self, stage: str):
        if stage == "fit":
            self.shape2idx = self.trainer.datamodule.train_dataset.shape2idx
            self._init_latent()

        print(self.shape2idx)

    def _init_latent(self):
        self.lat_vecs = nn.Embedding(
            len(self.shape2idx), self.model.lat_vecs.weight.shape[1]  # lat vec size
        )
        if self.hparams["emp_init"]:
            mean = self.model.lat_vecs.weight.mean()
            var = self.model.lat_vecs.weight.var()
            torch.nn.init.normal_(self.lat_vecs.weight.data, mean, var)
        else:
            std_lat_vec = 1.0 / math.sqrt(self.model.lat_vecs.weight.shape[1])
            torch.nn.init.normal_(self.lat_vecs.weight.data, 0.0, std_lat_vec)
        # save some GPU space
        self.model.lat_vecs = None

    def forward(self, x):
        return self.model(x)

    def predict(self, x, shapenet_idx):
        lat_vec = self.lat_vecs(shapenet_idx).expand(x.shape[1], -1).unsqueeze(0)
        return self.model.predict((x, lat_vec))

    def training_step(self, batch, batch_idx):
        self.model.eval()
        xyz = batch["xyz"]
        y = batch["sd"].flatten()
        lat_vec = self.lat_vecs(batch["idx"])

        y_hat = self.forward((xyz, lat_vec)).flatten()

        if self.hparams["clamp"]:
            y_hat = torch.clamp(
                y_hat, -self.hparams["clamp_val"], self.hparams["clamp_val"]
            )
            y = torch.clamp(y, -self.hparams["clamp_val"], self.hparams["clamp_val"])

        l1_loss = self.loss(y_hat, y)
        self.log("opt/l1_loss", l1_loss, on_step=True, on_epoch=True)
        if self.hparams["reg_loss"]:
            reg_loss = self.hparams["sigma"] * torch.mean(
                torch.linalg.norm(lat_vec, dim=2).flatten()
            )
            loss = l1_loss + reg_loss
            self.log("opt/reg_loss", reg_loss, on_step=True, on_epoch=True)
        else:
            loss = l1_loss
        self.log("opt/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optim = self.hparams["optim"](self.lat_vecs.parameters())
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optim)
            self.save_lr = scheduler.get_lr()
            return {"optimizer": optim, "lr_scheduler": scheduler}
        return optim

    def validation_step(self, batch, batch_idx):
        if len(batch["shapenet_idx"]) > 1:
            raise ValueError("Make sure that the batch_size for validation loader is 1")
        idx = batch["shapenet_idx"][0]
        mesh = self._get_obj(idx)
        chamfer = compute_chamfer_distance(batch["pointcloud"], mesh)
        self.log("val/chamfer", chamfer, on_epoch=True)

        self.idx2chamfer["idx"] = chamfer

    def _get_obj(self, shapenet_idx):
        device = self.model.device
        shapenet_idx_num = torch.tensor(
            [self.shape2idx[shapenet_idx]], device=self.device
        ).int()
        chunck_size = 500_000
        # hparams
        grid_vals = torch.arange(-1, 1, float(2 / self.resolution))
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
            sd = self.predict(_xyz, shapenet_idx_num).squeeze().cpu().numpy()
            sd_list.append(sd)
        sd = np.concatenate(sd_list)
        sd_r = sd.reshape(self.resolution, self.resolution, self.resolution)

        verts, faces, _, _ = marching_cubes(sd_r, level=0.0)

        x_max = np.array([1, 1, 1])
        x_min = np.array([-1, -1, -1])
        verts = verts * ((x_max - x_min) / (self.resolution)) + x_min

        # Create a trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        if self.hparams["save_obj"]:
            # Save the mesh as an OBJ file
            mesh.export(f"{self.hparams['save_obj_path']}/{shapenet_idx}.obj")

        return mesh


# Intendet to be used for one object at a Time, whereas the validator can optimize
# multiple ones
class DeepSDFOptimizer(L.LightningModule):
    # TODO
    # [ ] send to correct device automatically (self.latent is the issue)
    # [ ] check out why chair with prior_idx 37 is weird
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
        resolution: int = 256,
        emp_init: bool = False,
        save_obj: bool = False,
        save_obj_path: str = "",
        prior_idx: int = -1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["loss"])

        self._load_model()

        self.loss = loss

        self.resolution = resolution
        self.idx2chamfer = dict()

        if self.hparams["save_obj"] and not os.path.exists(
            self.hparams["save_obj_path"]
        ):
            os.mkdir(self.hparams["save_obj_path"])

    def _load_model(self):
        self.model = DeepSDF.load_from_checkpoint(
            self.hparams["ckpt_path"], map_location="cpu"
        )
        self.model.freeze()

    # TODO unnecessary right now, not calling setup at all
    def setup(self, stage: str):
        if stage == "fit":
            self.shape2idx = self.trainer.datamodule.train_dataset.shape2idx
            self._init_latent()

    def _init_latent(self):
        if self.hparams["prior_idx"] >= 0:
            # using a prior
            self.latent = self.model.lat_vecs(
                torch.tensor([self.hparams["prior_idx"]])
            ).detach()
            self.latent.requires_grad_()
        else:
            mean = self.model.lat_vecs.weight.mean(0)
            std = self.model.lat_vecs.weight.std(0)
            self.latent = torch.normal(mean.detach(), std.detach()).cuda()
            self.latent.requires_grad_()

        self.model.lat_vecs = None

    def forward(self, x):
        lat_vec = self.latent.expand(x.shape[1], -1).unsqueeze(0)
        return self.model((x, lat_vec))

    def predict(self, x):
        lat_vec = self.latent.expand(x.shape[1], -1).unsqueeze(0)
        return self.model.predict((x, lat_vec))

    def training_step(self, batch, batch_idx):
        self.model.eval()
        xyz = batch["xyz"]
        y = batch["sd"].flatten()

        y_hat = self.forward(xyz).flatten()

        if self.hparams["clamp"]:
            y_hat = torch.clamp(
                y_hat, -self.hparams["clamp_val"], self.hparams["clamp_val"]
            )
            y = torch.clamp(y, -self.hparams["clamp_val"], self.hparams["clamp_val"])

        l1_loss = self.loss(y_hat, y)
        self.log("opt/l1_loss", l1_loss, on_step=True, on_epoch=True)
        if self.hparams["reg_loss"]:
            # TODO will probably throw an error
            reg_loss = self.hparams["sigma"] * torch.linalg.norm(self.latent, dim=-1)
            loss = l1_loss + reg_loss
            self.log("opt/reg_loss", reg_loss, on_step=True, on_epoch=True)
        else:
            loss = l1_loss
        self.log("opt/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optim = self.hparams["optim"]([self.latent])
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optim)
            self.save_lr = scheduler.get_lr()
            return {"optimizer": optim, "lr_scheduler": scheduler}
        return optim

    def validation_step(self, batch, batch_idx):
        if len(batch["shapenet_idx"]) > 1:
            raise ValueError("Make sure that the batch_size for validation loader is 1")
        idx = batch["shapenet_idx"][0]
        mesh = self._get_obj(idx)
        chamfer = compute_chamfer_distance(batch["pointcloud"], mesh)
        self.log("val/chamfer", chamfer, on_epoch=True)

        self.idx2chamfer["idx"] = chamfer

    def get_obj(self, device="cuda"):
        self.to(device)
        chunck_size = 500_000
        # hparams
        grid_vals = torch.arange(-1, 1, float(2 / self.resolution))
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
        sd_r = sd.reshape(self.resolution, self.resolution, self.resolution)

        verts, faces, _, _ = marching_cubes(sd_r, level=0.0)

        x_max = np.array([1, 1, 1])
        x_min = np.array([-1, -1, -1])
        verts = verts * ((x_max - x_min) / (self.resolution)) + x_min

        # Create a trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        if self.hparams["save_obj"]:
            # Save the mesh as an OBJ file
            print("exporting")
            mesh.export(
                f"{self.hparams['save_obj_path']}/{self.trainer.max_epochs}-test.obj"
            )

        return mesh


class DeepSDFRenderOptimizer(L.LightningModule):
    # TODO
    # [ ] send to correct device automatically (self.latent is the issue)
    # [ ] check out why chair with prior_idx 37 is weird
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
        resolution: int = 256,
        emp_init: bool = False,
        save_obj: bool = False,
        save_obj_path: str = "",
        prior_idx: int = -1,
        n_render_steps: int = 100,
        surface_eps: float = 1e-4,
        sphere_eps: float = 3e-2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["loss"])

        self._load_model()

        self.loss = loss

        self.resolution = resolution
        self.idx2chamfer = dict()

        if self.hparams["save_obj"] and not os.path.exists(
            self.hparams["save_obj_path"]
        ):
            os.mkdir(self.hparams["save_obj_path"])

    def _load_model(self):
        self.model = DeepSDF.load_from_checkpoint(
            self.hparams["ckpt_path"], map_location="cpu"
        )
        self.model.freeze()

    # TODO unnecessary right now, not calling setup at all
    def setup(self, stage: str):
        if stage == "fit":
            # TODO replace this with a camera settigns npz file that can be used to obtain all important parameters: height, widht, focal, sphere_eps, surface_eps, etc
            # self.shape2idx = self.trainer.datamodule.train_dataset.shape2idx
            self._init_latent()

    def _init_latent(self):
        if self.hparams["prior_idx"] >= 0:
            # using a prior
            self.latent = self.model.lat_vecs(
                torch.tensor([self.hparams["prior_idx"]])
            ).detach()
            self.latent.requires_grad_()
        else:
            mean = self.model.lat_vecs.weight.mean(0)
            std = self.model.lat_vecs.weight.std(0)
            self.latent = torch.normal(mean.detach(), std.detach()).cuda()
            self.latent.requires_grad_()

        self.model.lat_vecs = None

    def forward(self, x):
        lat_vec = self.latent.expand(x.shape[1], -1).unsqueeze(0)
        return self.model((x, lat_vec))

    def predict(self, x):
        lat_vec = self.latent.expand(x.shape[1], -1).unsqueeze(0)
        return self.model.predict((x, lat_vec))

    def training_step(self, batch, batch_idx):
        self.model.eval()
        y = batch["render"]
        intersection = batch["intersection"]
        mask = batch["mask"]
        ray_direction = batch["ray_direction"]

        normals = self._render_normal(intersection, mask, ray_direction)

        mask = intersection.norm(dim=-1) > (1 + self.hparams["sphere_eps"] * 2)

        normals[mask.squeeze()] = torch.tensor([1, 1, 1], device=self.device).float()

        normals = normalize(normals)
        normals = normals.view(256, 256, 3).transpose(0, 1)
        y_hat = ((normals + 1) / 2) * 255

        l2_loss = self.loss(y_hat, y.float().squeeze())

        self.logger.log_image(key="normal", images=[y_hat.detach().cpu().numpy()])
        self.log("opt/l1_loss", l2_loss, on_step=True, on_epoch=True)
        if self.hparams["reg_loss"]:
            # TODO will probably throw an error
            reg_loss = self.hparams["sigma"] * torch.linalg.norm(self.latent, dim=-1)
            loss = l2_loss + reg_loss
            self.log("opt/reg_loss", reg_loss, on_step=True, on_epoch=True)
        else:
            loss = l2_loss
            self.log("debug/reg_norm", torch.linalg.norm(self.latent, dim=-1))
        self.log("opt/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _render_normal(self, intersection, mask, ray_direction):
        # TODO consider moving surface and sdfs to dataloader
        surface = torch.zeros_like(intersection, device=self.device)
        sdfs = torch.zeros(intersection.shape[1], device=self.device).half()
        for i in range(self.hparams["n_render_steps"]):
            sdf = self.predict(intersection[:, ~mask.squeeze()])
            sdfs[~mask.squeeze()] = sdf.squeeze()
            surface_mask = (sdfs < self.hparams["surface_eps"]) | (
                intersection.norm(dim=2) > (1 + self.hparams["sphere_eps"] * 2)
            )
            inv_acceleration = min(i / 10, 1)
            intersection[:, ~surface_mask.squeeze()] = (
                intersection + ray_direction * sdfs.unsqueeze(1) * inv_acceleration
            )[:, ~surface_mask.squeeze()]

        surface[:, surface_mask.squeeze()] = intersection[:, surface_mask.squeeze()]

        # Differentiable Part starts here

        intersection.requires_grad_()
        actual_surface_mask = intersection.norm(dim=2) > (
            1 + self.hparams["sphere_eps"] * 2
        )
        # normals = torch.ones_like(intersection)
        # normals.requires_grad_()
        # inp = intersection[:, ~actual_surface_mask.squeeze()]
        out = self.forward(intersection)
        # https://discuss.pytorch.org/t/what-determines-if-torch-autograd-grad-output-has-requires-grad-true/17104
        normals = torch.autograd.grad(
            outputs=out,
            inputs=intersection,
            grad_outputs=torch.ones_like(out),
            retain_graph=True,
            create_graph=True,
        )[0]
        return normals.squeeze()

    def configure_optimizers(self):
        optim = self.hparams["optim"]([self.latent])
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optim)
            self.save_lr = scheduler.get_lr()
            return {"optimizer": optim, "lr_scheduler": scheduler}
        return optim

    # def validation_step(self, batch, batch_idx):
    #     if len(batch["shapenet_idx"]) > 1:
    #         raise ValueError("Make sure that the batch_size for validation loader is 1")
    #     idx = batch["shapenet_idx"][0]
    #     mesh = self._get_obj(idx)
    #     chamfer = compute_chamfer_distance(batch["pointcloud"], mesh)
    #     self.log("val/chamfer", chamfer, on_epoch=True)

    #     self.idx2chamfer["idx"] = chamfer

    def get_obj(self, device="cuda"):
        self.to(device)
        chunck_size = 500_000
        # hparams
        grid_vals = torch.arange(-1, 1, float(2 / self.resolution))
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
        sd_r = sd.reshape(self.resolution, self.resolution, self.resolution)

        verts, faces, _, _ = marching_cubes(sd_r, level=0.0)

        x_max = np.array([1, 1, 1])
        x_min = np.array([-1, -1, -1])
        verts = verts * ((x_max - x_min) / (self.resolution)) + x_min

        # Create a trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        if self.hparams["save_obj"]:
            # Save the mesh as an OBJ file
            print("exporting")
            mesh.export(
                f"{self.hparams['save_obj_path']}/{self.trainer.max_epochs}-test.obj"
            )

        return mesh
