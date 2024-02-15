from typing import Callable, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from lib.data.dataset.latent_encoder import LatentEncoderDataset
from lib.data.metainfo import MetaInfo


class LatentEncoderDataModule(LightningDataModule):
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
        train_sampler: Optional[Sampler] = None,
        eval_sampler: Optional[Sampler] = None,
        dataset: Optional[LatentEncoderDataset] = None,
        sketch_transform: Optional[Callable] = None,
        normal_transform: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str):
        data_dir = self.hparams["data_dir"]
        if stage in ["fit", "all"]:
            self.train_metainfo = MetaInfo(data_dir=data_dir, split="train_latent")
            self.train_dataset = self.hparams["dataset"](
                data_dir=data_dir,
                split="train_latent",
            )
        if stage in ["validate", "fit", "all"]:
            self.val_metainfo = MetaInfo(data_dir=data_dir, split="val_latent")
            self.val_dataset = self.hparams["dataset"](
                data_dir=data_dir,
                split="val_latent",
            )

    def train_dataloader(self) -> DataLoader:
        self.train_metainfo.load_snn()
        labels = self.train_metainfo.snn_labels
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
        self.val_metainfo.load_snn()
        labels = self.val_metainfo.snn_labels
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
