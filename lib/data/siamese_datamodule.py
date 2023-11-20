from typing import Callable, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.sampler import Sampler
from torchvision import transforms

from lib.data.metainfo import MetaInfo
from lib.data.siamese_dataset import SiameseDatasetBase


def chunk_collate_fn(batch):
    batch = default_collate(batch)
    return {
        "sketch": batch["sketch"].flatten(0, 1),
        "image": batch["image"].flatten(0, 1),
        "label": batch["label"].flatten(0, 1),
    }


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
        shuffle: bool = True,
        # dataset
        sampler: Optional[Sampler] = None,
        dataset: Optional[SiameseDatasetBase] = None,
        collate_fn: Optional[Callable] = None,
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

    def build_sampler(self, split: str = "train"):
        sampler = None
        if self.hparams["sampler"]:
            metainfo = MetaInfo(data_dir=self.hparams["data_dir"], split=split)
            sampler = self.hparams["sampler"](labels=metainfo.labels)
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
            sampler=self.build_sampler("train"),
            collate_fn=self.hparams["collate_fn"],
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            sampler=self.build_sampler("val"),
            collate_fn=self.hparams["collate_fn"],
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            sampler=self.build_sampler("test"),
            collate_fn=self.hparams["collate_fn"],
        )
