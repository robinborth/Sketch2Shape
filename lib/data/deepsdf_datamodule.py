from typing import Optional

from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from lib.data.deepsdf_dataset import DeepSDFDataset, DeepSDFLatentOptimizerDataset


class DeepSDFDataModule(LightningDataModule):
    def __init__(
        self,
        # settings
        data_dir: str = "data/",
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
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str) -> None:
        self.train_dataset = self.hparams["dataset"](
            data_dir=self.hparams["data_dir"],
            split="train",
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


class DeepSDFLatentOptimizationDataModule(LightningDataModule):
    def __init__(
        self,
        # settings
        data_dir: str = "data/",
        obj_id: str = "obj_id",
        subsample: int = 16384,
        half: bool = False,
        # training
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = True,
        persistent_workers: bool = False,
        shuffle: bool = False,
        # dataset
        dataset: Optional[DeepSDFLatentOptimizerDataset] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str) -> None:
        self.dataset = self.hparams["dataset"](
            data_dir=self.hparams["data_dir"],
            obj_id=self.hparams["obj_id"],
            subsample=self.hparams["subsample"],
            half=self.hparams["half"],
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            shuffle=self.hparams["shuffle"],
        )


class RenderedSDFDataModule(LightningDataModule):
    def __init__(
        self,
        # settings
        data_dir: str = "data/",
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
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str) -> None:
        self.train_dataset = self.hparams["dataset"](data_dir=self.hparams["data_dir"])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=1,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            shuffle=self.hparams["shuffle"],
        )
