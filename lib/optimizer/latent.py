import re

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from tqdm import tqdm

from lib.data.metainfo import MetaInfo
from lib.data.transforms import BaseTransform, DilateSketch
from lib.models.deepsdf import DeepSDF
from lib.models.loss import Loss


class LatentOptimizer(LightningModule):
    def __init__(
        self,
        # base settings
        data_dir: str = "/data",
        loss_ckpt_path: str = "siamese.ckpt",
        deepsdf_ckpt_path: str = "deepsdf.ckpt",
        optimizer=None,
        scheduler=None,
        # init settings: random, mean, prior, prior(idx), retrieval, latent
        latent_init: str = "mean",
        # regularization settings: none, prior, retrieval, latent
        reg_loss: str = "none",
        reg_weight: float = 1e-5,
        # retrieval settings for init and prior
        prior_obj_id: str = "",
        prior_view_id: int = 11,
        retrieval_k: int = 16,
        # mesh settings
        mesh_resolution: int = 128,
        mesh_chunk_size: int = 65536,
        # rendering settings
        n_render_steps: int = 100,
        clamp_sdf: float = 0.1,
        step_scale: float = 1.0,
        surface_eps: float = 1e-03,
        sphere_eps: float = 1e-01,
        normal_eps: float = 5e-03,
        # image logger settings
        log_images: bool = True,
        capture_rate: int = 30,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.metainfo = MetaInfo(data_dir=data_dir)
        # TODO fix the normal transforms and the init with other
        # sketches e.g. handdrawn and grayscale
        self.transforms = BaseTransform(transforms=[DilateSketch(kernel_size=5)])

        # init deepsdf
        self.deepsdf = DeepSDF.load_from_checkpoint(
            deepsdf_ckpt_path,
            strict=True,
            # mesh settings
            mesh_resolution=mesh_resolution,
            mesh_chunk_size=mesh_chunk_size,
            # rendering settings
            n_render_steps=n_render_steps,
            clamp_sdf=clamp_sdf,
            step_scale=step_scale,
            surface_eps=surface_eps,
            sphere_eps=sphere_eps,
            normal_eps=normal_eps,
        )
        self.deepsdf.freeze()
        self.deepsdf.eval()
        self.deepsdf.create_camera()

        self.loss = Loss.load_from_checkpoint(loss_ckpt_path)
        self.loss.freeze()
        self.loss.eval()

        # retrieve the latent code based on the sketch
        if latent_init.startswith("retrieval") or reg_loss.startswith("retrieval"):
            self.init_retrieval_latents(
                obj_id=prior_obj_id,
                view_id=prior_view_id,
                k=retrieval_k,
            )
        self.init_latent(
            name="latent",
            latent_init=latent_init,
            obj_id=prior_obj_id,
        )

    def get_latent(self, latent_init: str = "mean", obj_id: str = ""):
        if latent_init == "prior":
            prior_idx = self.metainfo.obj_id_to_label(obj_id=obj_id)
            return self.deepsdf.get_latent(prior_idx)

        if match := re.match(r"prior\((\d+)\)", latent_init):
            prior_idx = int(match.group(1))
            return self.deepsdf.get_latent(prior_idx)

        if latent_init == "retrieval":
            return self.retrieval_latents.mean(0)

        if latent_init == "latent":
            assert self.loss.support_latent
            label = self.metainfo.obj_id_to_label(obj_id)
            view_id = self.hparams["prior_view_id"]
            sketch = self.metainfo.load_image(label, view_id, 0)  # sketch
            loss_input = self.transforms(sketch)[None, ...].to(self.loss.device)
            return self.loss.embedding(loss_input)[0]

        if latent_init == "mean":
            return self.deepsdf.get_latent(-1)

        if latent_init == "random":
            return self.deepsdf.get_latent(-2)

        raise NotImplementedError("The current settings are not supported!")

    def init_latent(
        self,
        latent_init: str = "mean",
        obj_id: str = "",
        name: str = "latent",
    ):
        latent = self.get_latent(latent_init=latent_init, obj_id=obj_id)
        self.register_buffer(name, latent)

    def init_retrieval_latents(
        self,
        obj_id: str,
        view_id: int = 11,
        k: int = 16,
    ):
        device = self.loss.device
        obj_id_label = int(self.metainfo.obj_id_to_label(obj_id))
        sketch = self.metainfo.load_image(obj_id_label, view_id, 0)  # sketch
        loss_input = self.transforms(sketch)[None, ...].to(device)
        sketch_emb = self.loss.embedding(loss_input, mode="sketch")

        # get the loss from all objects in the train dataset
        metainfo = MetaInfo(data_dir=self.hparams["data_dir"], split="train_latent")
        _loss = []
        for obj_id in tqdm(metainfo.obj_ids):
            label = int(metainfo.obj_id_to_label(obj_id))
            normal = metainfo.load_image(label, view_id, 1)  # normal
            loss_input = self.transforms(normal)[None, ...].to(device)
            normal_emb = self.loss.embedding(loss_input, mode="normal")
            loss = self.loss.compute(sketch_emb, normal_emb)
            _loss.append(loss)
        self.shape_loss = torch.concatenate(_loss)

        # don't include the latent code that we want to optimize for
        idx = torch.argsort(self.shape_loss)
        retrieval_idx = idx[idx != obj_id_label][:k]
        retrieval_latents = self.deepsdf.lat_vecs.weight[retrieval_idx]
        self.register_buffer("retrieval_latents", retrieval_latents)
        self.retrieval_idx = retrieval_idx

    def forward(self, points: torch.Tensor, mask=None):
        return self.deepsdf(points=points, latent=self.latent, mask=mask)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Please provide the optimization implementation.")

    def on_train_epoch_start(self):
        self.deepsdf.eval()
        self.loss.eval()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx % self.hparams["capture_rate"] == 0:
            self.capture_camera_frame()

    def on_before_optimizer_step(self, optimizer):
        """
        In order to debug vanishing gradient problems, we compute the 2-norm for each
        layer. If using mixed precision, the gradients are already unscaled here.
        """
        norm = self.latent.grad.norm(p=2)
        self.log("optimize/grad_norm", norm)

    def configure_optimizers(self):
        self.latent = self.latent.detach()
        self.latent.requires_grad = True
        optimizer = self.hparams["optimizer"]([self.latent])
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def to_mesh(self):
        return self.deepsdf.to_mesh(self.latent)

    def capture_camera_frame(self, mode="normal"):
        image = self.deepsdf.capture_camera_frame(self.latent, mode=mode)
        self.log_image("camera_frame", image)
        return image

    def log_image(self, key: str, image: torch.Tensor):
        image = image.detach().cpu().numpy()
        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(key, [image])  # type: ignore
