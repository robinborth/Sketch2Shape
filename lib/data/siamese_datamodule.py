from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms

from lib.data.metainfo import MetaInfo
from lib.data.siamese_dataset import SiameseDatasetBase


class SiameseDataModule(LightningDataModule):
    def __init__(
        self,
        # paths
        data_dir: str = "data/",
        dataset_splits_path: str = "data/dataset_splits.csv",
        sketch_image_pairs_path: str = "data/dataset_splits.csv",
        # training
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = True,
        persistent_workers: bool = False,
        sampler: Optional[Sampler] = None,
        # dataset
        dataset: Optional[SiameseDatasetBase] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = self.hparams["dataset"](
                stage="train",
                data_dir=self.hparams["data_dir"],
                transforms=self.transforms,
            )
            self.val_dataset = self.hparams["dataset"](
                stage="val",
                data_dir=self.hparams["data_dir"],
                transforms=self.transforms,
            )
        elif stage == "validate":
            self.val_dataset = self.hparams["dataset"](
                stage="val",
                data_dir=self.hparams["data_dir"],
                transforms=self.transforms,
            )
        elif stage == "test":
            self.test_dataset = self.hparams["dataset"](
                stage="test",
                data_dir=self.hparams["data_dir"],
                transforms=self.transforms,
            )

    def train_dataloader(self) -> DataLoader:
        metainfo = MetaInfo(data_dir=self.hparams["data_dir"], split="train")
        sampler = self.hparams["sampler"](labels=metainfo.labels)
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        metainfo = MetaInfo(data_dir=self.hparams["data_dir"], split="val")
        sampler = self.hparams["sampler"](labels=metainfo.labels)
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
        metainfo = MetaInfo(data_dir=self.hparams["data_dir"], split="test")
        sampler = self.hparams["sampler"](labels=metainfo.labels)
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            sampler=sampler,
        )
