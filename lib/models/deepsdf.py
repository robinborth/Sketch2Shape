import math
import os

import numpy as np
import torch
import torch.nn as nn
import trimesh
from lightning import LightningModule
from skimage.measure import marching_cubes

from lib.evaluate import compute_chamfer_distance


class DeepSDF(LightningModule):
    def __init__(
        self,
        loss: torch.nn.Module,
        decoder_optimizer: torch.optim.Optimizer,
        latents_optimizer: torch.optim.Optimizer,
        latent_size: int = 512,
        num_hidden_layers: int = 8,
        latent_vector_size: int = 256,
        num_scenes: int = 1,  # TODO rename num_latent_vectors
        clamp: bool = True,
        clamp_val: float = 0.1,
        reg_loss: bool = True,
        sigma: float = 1e-4,  # TODO rename reg_weight
        skip_connection: list[int] = [4],
        dropout: bool = True,  # TODO drop and only use dropout_p
        dropout_p: float = 0.2,  # TODO rename dropout
        dropout_latent: bool = False,  # TODO drop deprecated
        dropout_latent_p: float = 0.2,  # TODO drop deprecated
        weight_norm: bool = False,
        decoder_scheduler=None,
        latents_scheduler=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False

        self.loss = loss
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
            layers.append(nn.Sequential(layer, nn.ReLU(), nn.Dropout(p=dropout_p)))

        # # output layer and final deepsdf backbone
        layers.append(nn.Sequential(nn.Linear(latent_size, 1), nn.Tanh()))
        self.decoder = nn.Sequential(*layers)

        # latent vectors
        self.lat_vecs = nn.Embedding(num_scenes, latent_vector_size)
        std_lat_vec = 1.0 / math.sqrt(latent_vector_size)
        torch.nn.init.normal_(self.lat_vecs.weight.data, 0.0, std_lat_vec)

    def forward(self, points: torch.Tensor, latent: torch.Tensor):
        """The forward pass of the deepsdf model.

        Args:
            points (torch.Tensor): The points of dim (B, N, 3) or (N, 3).
            latent (torch.Tensor): The latent code of dim (B, L) or (L).

        Returns:
            torch.Tensor: The sdf values of dim (B, N)
        """
        N, L = points.shape[-2], latent.shape[-1]
        if len(latent.shape) == 1:
            latent = latent.unsqueeze(-2).expand(N, L)
        else:
            latent = latent.unsqueeze(-2).expand(-1, N, L)

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

        gt_sdf = batch["gt_sdf"]  # (B, N)
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
            reg_loss *= self.hparams["sigma"]
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


# custom validation loop
class DeepSDFValidator(LightningModule):
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
class DeepSDFOptimizer(LightningModule):
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

        if self.hparams["save_obj"] and not os.path.exists(
            self.hparams["save_obj_path"]
        ):
            os.mkdir(self.hparams["save_obj_path"])

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

    def validation_step(self, batch, batch_idx):
        if len(batch["shapenet_idx"]) > 1:
            raise ValueError("Make sure that the batch_size for validation loader is 1")
        idx = batch["shapenet_idx"][0]
        mesh = self._get_obj(idx)
        chamfer = compute_chamfer_distance(batch["pointcloud"], mesh)
        self.log("val/chamfer", chamfer, on_epoch=True)

        self.idx2chamfer["idx"] = chamfer
