from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

from lib.data.metainfo import MetaInfo


class LossDataModule(LightningDataModule):
    def __init__(
        self,
        # settings
        data_dir: str = "data/",
        modes: list[int] = [0, 1],
        latent: bool = False,
        # training
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = True,
        persistent_workers: bool = False,
        shuffle: bool = True,
        # dataset
        train_sampler: Optional[Sampler] = None,
        eval_sampler: Optional[Sampler] = None,
        dataset: Optional[Dataset] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str):
        data_dir = self.hparams["data_dir"]
        modes = self.hparams["modes"]
        if stage in ["fit", "all"]:
            split = "train_latent" if self.hparams["latent"] else "train"
            self.train_metainfo = MetaInfo(data_dir=data_dir, split=split)
            self.train_dataset = self.hparams["dataset"](
                data_dir=data_dir,
                split="train",
                modes=modes,
            )
        if stage in ["validate", "fit", "all"]:
            split = "val_latent" if self.hparams["latent"] else "val"
            self.val_metainfo = MetaInfo(data_dir=data_dir, split=split)
            self.val_dataset = self.hparams["dataset"](
                data_dir=data_dir,
                split="val",
                modes=modes,
            )
        if stage in ["test", "all"]:
            self.test_metainfo = MetaInfo(data_dir=data_dir, split="test")
            self.test_dataset = self.hparams["dataset"](
                data_dir=data_dir,
                split="test",
                modes=modes,
            )

    def train_dataloader(self) -> DataLoader:
        self.train_metainfo.load_loss(modes=self.hparams["modes"])
        labels = self.train_metainfo.loss_labels
        sampler = self.hparams["train_sampler"](labels=labels)
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            shuffle=self.hparams["shuffle"],
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        self.val_metainfo.load_loss(modes=self.hparams["modes"])
        labels = self.val_metainfo.loss_labels
        sampler = self.hparams["eval_sampler"](labels=labels)
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            sampler=sampler,
        )

    def test_dataloader(self) -> DataLoader:
        self.test_metainfo.load_loss(modes=self.hparams["modes"])
        labels = self.test_metainfo.loss_labels
        sampler = self.hparams["eval_sampler"](labels=labels)
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            sampler=sampler,
        )
