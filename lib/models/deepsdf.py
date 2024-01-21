import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from lightning import LightningModule
from skimage.measure import marching_cubes
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.eval.chamfer_distance import ChamferDistanceMetric
from lib.eval.clip_score import CLIPScoreMetric
from lib.eval.earth_movers_distance import EarthMoversDistanceMetric
from lib.eval.frechet_inception_distance import FrechetInceptionDistanceMetric

############################################################
# DeepSDF Training
############################################################


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


############################################################
# DeepSDF Latent Optimizier Base
# This is used for all the optimization and evalution parts
# in the project, e.g. also for the rendering modules.
############################################################


class DeepSDFLatentOptimizerBase(LightningModule):
    def __init__(
        self,
        ckpt_path: str = "best.ckpt",
        prior_idx: int = -1,
        reg_loss: bool = True,
        reg_weight: float = 1e-05,
        optimizer=None,
        scheduler=None,
        resolution: int = 128,
        chunk_size: int = 65536,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # deepsdf options
        self.min_val = -1
        self.max_val = 1

        # init model
        self.model = DeepSDF.load_from_checkpoint(self.hparams["ckpt_path"])
        self.model.freeze()

        # init latent either by using a pretrained one ore the mean of the pretrained
        if self.hparams["prior_idx"] >= 0:
            idx = torch.tensor([self.hparams["prior_idx"]])
            latent = self.model.lat_vecs(idx.to(self.model.device)).squeeze()
        else:
            latent = self.model.lat_vecs.weight.mean(0)
        self.register_buffer("latent", latent)
        self.mesh: o3d.geometry.TriangleMesh = None

        #  metrics
        self.chamfer_distance = ChamferDistanceMetric()
        # TODO add the other metrics here

    def forward(self, points: torch.Tensor, mask=None):
        return self.model(points=points, latent=self.latent, mask=mask)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Please provide the optimization implementation.")

    def test_step(self, batch, batch_idx):
        gt_surface_samples = batch["surface_samples"].detach().cpu().numpy().squeeze()
        mesh = self.to_mesh(self.hparams["resolution"], self.hparams["chunk_size"])
        chamfer = self.chamfer_distance(mesh, gt_surface_samples)
        self.log("val/chamfer", chamfer)

    def configure_optimizers(self):
        self.latent = self.latent.detach()
        self.latent.requires_grad = True
        optimizer = self.hparams["optimizer"]([self.latent])
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def to_mesh(
        self,
        resolution: int = 256,
        chunk_size: int = 65536,
    ) -> o3d.geometry.TriangleMesh:
        self.model.eval()
        min_val, max_val = self.min_val, self.max_val
        # TODO only sample in the unit sphere, the other points should be positive
        grid_vals = torch.linspace(min_val, max_val, resolution)
        xs, ys, zs = torch.meshgrid(grid_vals, grid_vals, grid_vals, indexing="ij")
        points = torch.stack((xs.ravel(), ys.ravel(), zs.ravel())).transpose(1, 0)

        loader = DataLoader(points, batch_size=chunk_size)  # type: ignore
        sd = []
        for points in tqdm(iter(loader), total=len(loader)):
            points = points.to(self.model.device)
            sd_out = self.forward(points).detach().cpu().numpy()
            sd.append(sd_out)
        sd_cube = np.concatenate(sd).reshape(resolution, resolution, resolution)

        verts, faces, _, _ = marching_cubes(sd_cube, level=0.0)
        verts = verts * ((max_val - min_val) / resolution) + min_val

        # override the current mesh
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(verts)
        self.mesh.triangles = o3d.utility.Vector3iVector(faces)

        return self.mesh


############################################################
# DeepSDF Latent Optimization (Validation)
############################################################


class DeepSDFLatentOptimizer(DeepSDFLatentOptimizerBase):
    def __init__(
        self,
        loss: torch.nn.Module,
        clamp: bool = True,
        clamp_val: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model.lat_vecs = None
        self.loss = loss()

    def training_step(self, batch, batch_idx):
        gt_sdf = batch["sdf"].squeeze()  # (N)
        points = batch["points"].squeeze()  # (N, 3)

        sdf = self.forward(points=points)  # (N)

        if self.hparams["clamp"]:
            clamp_val = self.hparams["clamp_val"]
            sdf = torch.clamp(sdf, -clamp_val, clamp_val)
            gt_sdf = torch.clamp(gt_sdf, -clamp_val, clamp_val)

        l1_loss = self.loss(sdf, gt_sdf)
        self.log("train/l1_loss", l1_loss, on_step=True, on_epoch=True)

        reg_loss = torch.tensor(0).to(l1_loss)
        if self.hparams["reg_loss"]:
            reg_loss = torch.linalg.norm(self.latent, dim=-1).mean()
            reg_loss *= min(1, self.current_epoch / 100)
            reg_loss *= self.hparams["reg_weight"]
            self.log("train/reg_loss", reg_loss, on_step=True, on_epoch=True)

        loss = l1_loss + reg_loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss


############################################################
# Latent Code Traversal
############################################################


class DeepSDFLatentTraversal(DeepSDFLatentOptimizerBase):
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
        mesh = self.to_mesh(self.hparams["resolution"], self.hparams["chunk_size"])
        self.meshes.append(mesh)

        # restore the mean latent
        self.latent = latent
