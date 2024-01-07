from typing import Callable, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from lib.data.metainfo import MetaInfo
from lib.data.siamese_dataset import SiameseDataset


class SiameseDataModule(LightningDataModule):
    def __init__(
        self,
        # paths
        data_dir: str = "data/",
        # training
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = True,
        persistent_workers: bool = False,
        shuffle: bool = True,
        # dataset
        sampler: Optional[Sampler] = None,
        dataset: Optional[SiameseDataset] = None,
        transforms: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.metainfo = MetaInfo(data_dir=self.hparams["data_dir"])

    def setup(self, stage: str):
        data_dir = self.hparams["data_dir"]
        if stage in ["fit", "all"]:
            self.train_metainfo = MetaInfo(data_dir=data_dir, split="train")
            self.train_dataset = self.hparams["dataset"](
                metainfo=self.train_metainfo,
                transforms=self.hparams["transforms"],
            )
        if stage in ["validate", "fit", "all"]:
            self.val_metainfo = MetaInfo(data_dir=data_dir, split="val")
            self.val_dataset = self.hparams["dataset"](
                metainfo=self.val_metainfo,
                transforms=self.hparams["transforms"],
            )
        if stage in ["test", "all"]:
            self.test_metainfo = MetaInfo(data_dir=data_dir, split="test")
            self.test_dataset = self.hparams["dataset"](
                metainfo=self.test_metainfo,
                transforms=self.hparams["transforms"],
            )

    def build_sampler(self, metainfo: MetaInfo):
        sampler = None
        if self.hparams["sampler"]:
            sampler = self.hparams["sampler"](
                labels=metainfo.labels,
                length_before_new_iter=metainfo.pair_count,
            )
        return sampler

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            shuffle=self.hparams["shuffle"],
            sampler=self.build_sampler(self.train_metainfo),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            sampler=self.build_sampler(self.val_metainfo),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            sampler=self.build_sampler(self.test_metainfo),
        )
