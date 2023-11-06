import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

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
            self.train_dataset = instantiate(self.cfg.dataset, self.cfg, "train")
            self.val_dataset = instantiate(self.cfg.dataset, self.cfg, "val")
        elif stage == "validate":
            self.val_dataset = instantiate(self.cfg.dataset, self.cfg, "val")
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
