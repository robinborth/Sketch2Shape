from typing import Callable, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, default_collate


class LossTesterDataModule(LightningDataModule):
    def __init__(
        self,
        # paths
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
        dataset=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str):
        data_dir = self.hparams["data_dir"]
        modes = self.hparams["modes"]
        if stage in ["fit", "all"]:
            split = "train_latent" if self.hparams["latent"] else "train"
            self.train_dataset = self.hparams["dataset"](
                data_dir=data_dir,
                split=split,
                modes=modes,
            )
        if stage in ["validate", "fit", "all"]:
            split = "val_latent" if self.hparams["latent"] else "val"
            self.val_dataset = self.hparams["dataset"](
                data_dir=data_dir,
                split=split,
                modes=modes,
            )
        if stage in ["test", "all"]:
            self.test_dataset = self.hparams["dataset"](
                data_dir=data_dir,
                split="test",
                modes=modes,
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
