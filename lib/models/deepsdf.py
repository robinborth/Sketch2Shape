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
        layers.append(nn.Linear(3, cfg.embedding_size))
        layers.append(nn.ReLU())
        for _ in range(cfg.num_hidden_layers):
            layers.append(nn.Linear(cfg.embedding_size, cfg.embedding_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(cfg.embedding_size, 1))
        self.backbone = nn.Sequential(*layers)

    def forward(self, batch):
        out = {}
        out["sd"] = self.backbone(batch["xyz"])
        return out

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.loss(output["sd"], batch["sd"].reshape(-1, 1))
        self.log("train/loss", loss, prog_bar=True, batch_size=self.cfg.batch_size)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.cfg.learning_rate,
        )
