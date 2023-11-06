import lightning as L
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


class SDFBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout, dropout_p):
        super(SDFBlock, self).__init__()
        # self.linear = nn.utils.weight_norm(
        #     nn.Linear(in_features=in_features, out_features=out_features)
        # )
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = dropout
        self.dropout_p = dropout_p

    def forward(self, x):
        x = self.linear(x)
        if self.dropout:
            x = nn.functional.dropout(x, p=self.dropout_p)
        x = nn.functional.relu(x)
        return x


class DeepSDFModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(DeepSDFModel, self).__init__()
        self.cfg = cfg
        self._build_model()

    def _build_model(self):
        self.block1 = SDFBlock(
            # self.cfg.model.latent_vector_size + 3,
            3,
            self.cfg.model.latent_size,
            self.cfg.model.dropout,
            self.cfg.model.dropout_p,
        )
        self.block2 = SDFBlock(
            self.cfg.model.latent_size,
            self.cfg.model.latent_size,
            self.cfg.model.dropout,
            self.cfg.model.dropout_p,
        )
        # self.block3 = SDFBlock(
        #     self.cfg.model.latent_size,
        #     self.cfg.model.latent_size,
        #     self.cfg.model.dropout,
        #     self.cfg.model.dropout_p,
        # )
        # self.block4 = SDFBlock(
        #     self.cfg.model.latent_size,
        #     self.cfg.model.latent_size - self.cfg.model.latent_vector_size,
        #     self.cfg.model.dropout,
        #     self.cfg.model.dropout_p,
        # )
        # self.block5 = SDFBlock(
        #     self.cfg.model.latent_size,
        #     self.cfg.model.latent_size,
        #     self.cfg.model.dropout,
        #     self.cfg.model.dropout_p,
        # )
        # self.block6 = SDFBlock(
        #     self.cfg.model.latent_size,
        #     self.cfg.model.latent_size,
        #     self.cfg.model.dropout,
        #     self.cfg.model.dropout_p,
        # )
        # self.block7 = SDFBlock(
        #     self.cfg.model.latent_size,
        #     self.cfg.model.latent_size,
        #     self.cfg.model.dropout,
        #     self.cfg.model.dropout_p,
        # )
        self.block8 = nn.Sequential(nn.Linear(self.cfg.model.latent_size, 1))

    def forward(self, x):
        xyz, lat_vec = x
        # input = torch.hstack((lat_vec, xyz))
        input = xyz
        out = self.block1(input)
        out = self.block2(out)
        out = self.block3(out)
        # out = self.block4(out)

        # out = torch.hstack((out, lat_vec))
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        return out


class DeepSDF(L.LightningModule):
    def __init__(self, cfg: DictConfig, num_scenes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.loss = torch.nn.MSELoss(reduction="mean")
        # self.loss = torch.nn.L1Loss(reduction="mean")
        # self.decoder = DeepSDFModel(self.cfg)
        self.lat_vecs = torch.nn.Embedding(
            num_scenes, self.cfg.model.latent_vector_size
        )
        # torch.nn.init.normal_(self.lat_vecs.weight.data, 0.0, 0.01)
        self.decoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        ).double()

    def forward(self, x):
        return self.decoder(x)

    def _merge_latents(self, batch):
        # batch[0]: [x, y, z]
        # batch[1]: [sdf]
        # batch[2]: [shape_idx]
        lat_vec = self.lat_vecs(batch[2].int())
        xyz = batch[0]
        sd = batch[1]
        return xyz, lat_vec, sd

    def training_step(self, batch, batch_idx):
        # xyz, lat_vec, sd = self._merge_latents(batch)
        x = batch["xyz"]
        sd = batch["sd"].reshape(-1, 1)
        # x = (xyz, lat_vec)
        sd_hat = self.forward(x)

        if self.cfg.model.clamp:
            sd = torch.clamp(
                sd, min=-self.cfg.model.clamp_delta, max=self.cfg.model.clamp_delta
            )
            sd_hat = torch.clamp(
                sd_hat, min=-self.cfg.model.clamp_delta, max=self.cfg.model.clamp_delta
            )

        # SDF Loss
        sdf_loss = self.loss(sd_hat, sd)
        # Reg loss
        if self.cfg.model.reg_loss:
            reg_loss = torch.mean(torch.linalg.norm(x[1])) * self.cfg.model.sigma
            reg_loss /= x[0].shape[0]
            loss = sdf_loss + reg_loss
        else:
            loss = sdf_loss

        # logging
        self.log("train/loss", loss, prog_bar=True, batch_size=self.cfg.batch_size)
        if self.cfg.model.reg_loss:
            self.log("train/sdf_loss", sdf_loss)
            self.log(
                "train/reg_loss",
                reg_loss,
            )
        return loss

    def predict(self, xyz, lat_vec):
        """
        Inference Time optimization
        """
        x = (xyz, lat_vec)
        return self.decoder(xyz)

    def configure_optimizers(self):
        return torch.optim.Adam(
            [
                {"params": self.decoder.parameters(), "lr": self.cfg.model.lr_decoder},
                # {
                #     "params": self.lat_vecs.parameters(),
                #     "lr": self.cfg.model.lr_lat_vecs,
                # },
            ]
        )

        # optimizer_decoder = Adam(
        #     self.decoder.parameters(), lr=self.cfg.model.lr_decoder
        # )
        # optimizer_lat_vecs = Adam(
        #     self.lat_vecs.parameters(), lr=self.cfg.model.lr_lat_vecs
        # )

        # # Apply CosineAnnealingLR schedule to both optimizers
        # scheduler_decoder = CosineAnnealingLR(
        #     optimizer_decoder,
        #     T_max=self.cfg.trainer.max_epochs,
        #     eta_min=self.cfg.model.lr_decoder * 0.001,
        # )
        # scheduler_lat_vecs = CosineAnnealingLR(
        #     optimizer_lat_vecs,
        #     T_max=self.cfg.trainer.max_epochs,
        #     eta_min=self.cfg.model.lr_lat_vecs * 0.001,
        # )

        # return (
        #     {"optimizer": optimizer_decoder, "lr_scheduler": scheduler_decoder},
        #     {"optimizer": optimizer_lat_vecs, "lr_scheduler": scheduler_lat_vecs},
        # )
