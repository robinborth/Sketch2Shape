from typing import Callable, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, default_collate


class LossTesterDataModule(LightningDataModule):
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
        dataset=None,
        sketch_transform: Optional[Callable] = None,
        normal_transform: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str):
        self.train_dataset = self.hparams["dataset"](
            data_dir=self.hparams["data_dir"],
            split="train",
            sketch_transform=self.hparams["sketch_transform"],
            normal_transform=self.hparams["normal_transform"],
        )
        self.val_dataset = self.hparams["dataset"](
            data_dir=self.hparams["data_dir"],
            split="val",
            sketch_transform=self.hparams["sketch_transform"],
            normal_transform=self.hparams["normal_transform"],
        )
        self.test_dataset = self.hparams["dataset"](
            data_dir=self.hparams["data_dir"],
            split="test",
            sketch_transform=self.hparams["sketch_transform"],
            normal_transform=self.hparams["normal_transform"],
        )

    @staticmethod
    def collate_fn(batch):
        assert len(batch) == 1
        return default_collate(batch[0])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            collate_fn=self.collate_fn,
        )
