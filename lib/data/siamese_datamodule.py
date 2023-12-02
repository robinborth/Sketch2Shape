from typing import Callable, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from torchvision.models.resnet import ResNet18_Weights

from lib.data.metainfo import MetaInfo
from lib.data.siamese_dataset import SiameseDatasetBase


def chunk_collate_fn(batch):
    batch = default_collate(batch)
    return {
        "obj_id": batch["obj_id"],
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
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # self.transforms = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #     ]
        # )

    def setup(self, stage: str):
        if stage in ["fit", "all"]:
            self.train_metainfo = MetaInfo(
                data_dir=self.hparams["data_dir"],
                dataset_splits_path=self.hparams["dataset_splits_path"],
                sketch_image_pairs_path=self.hparams["sketch_image_pairs_path"],
                split="train",
            )
            self.train_dataset = self.hparams["dataset"](
                metainfo=self.train_metainfo,
                transforms=self.transforms,
            )
        elif stage in ["validate", "fit", "all"]:
            self.val_metainfo = MetaInfo(
                data_dir=self.hparams["data_dir"],
                dataset_splits_path=self.hparams["dataset_splits_path"],
                sketch_image_pairs_path=self.hparams["sketch_image_pairs_path"],
                split="val",
            )
            self.val_dataset = self.hparams["dataset"](
                metainfo=self.val_metainfo,
                transforms=self.transforms,
            )
        if stage in ["test", "all"]:
            self.test_metainfo = MetaInfo(
                data_dir=self.hparams["data_dir"],
                dataset_splits_path=self.hparams["dataset_splits_path"],
                sketch_image_pairs_path=self.hparams["sketch_image_pairs_path"],
                split="test",
            )
            self.test_dataset = self.hparams["dataset"](
                metainfo=self.test_metainfo,
                transforms=self.transforms,
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
            sampler=self.build_sampler(self.val_metainfo),
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
            sampler=self.build_sampler(self.test_metainfo),
            collate_fn=self.hparams["collate_fn"],
        )
