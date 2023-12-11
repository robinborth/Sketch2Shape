from typing import Optional

from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from lib.data.deepsdf_dataset import DeepSDFDataset


class DeepSDFDataModule(LightningDataModule):
    def __init__(
        self,
        # settings
        data_dir: str = "data/",
        load_ram: bool = True,
        subsample: int = 16384,
        half: bool = False,
        # training
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = True,
        persistent_workers: bool = False,
        shuffle: bool = False,
        # dataset
        dataset: Optional[DeepSDFDataset] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str) -> None:
        self.train_dataset = self.hparams["dataset"](
            data_dir=self.hparams["data_dir"],
            load_ram=self.hparams["load_ram"],
            subsample=self.hparams["subsample"],
            half=self.hparams["half"],
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            shuffle=self.hparams["shuffle"],
        )


class OneShapeSDFDataModule(LightningDataModule):
    def __init__(
        self,
        # settings
        path: str,
        subsample: int = 65536,
        # training
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = True,
        persistent_workers: bool = False,
        shuffle: bool = False,
        # dataset
        dataset: Optional[DeepSDFDataset] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str) -> None:
        self.dataset = self.hparams["dataset"](
            path=self.hparams["path"],
            subsample=self.hparams["subsample"],
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=1,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
        )
