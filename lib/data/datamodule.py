import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from lib.data.dataset import ShapeNetDataset
from lib.data.metainfo import MetaInfo


class ShapeNetDataModule(L.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.cfg = cfg

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = ShapeNetDataset(stage="train", cfg=self.cfg)
            self.val_dataset = ShapeNetDataset(stage="val", cfg=self.cfg)
        elif stage == "validate":
            self.val_dataset = ShapeNetDataset(stage="val", cfg=self.cfg)
        else:
            raise NotImplementedError()

    def train_dataloader(self) -> DataLoader:
        metainfo = MetaInfo(cfg=self.cfg, split="train")
        self.sampler = instantiate(self.cfg.sampler, metainfo.labels)
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            sampler=self.sampler,
            drop_last=self.cfg.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
