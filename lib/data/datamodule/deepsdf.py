from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DeepSDFDataModule(LightningDataModule):
    def __init__(
        self,
        # settings
        data_dir: str = "data/",
        chunk_size: int = 16384,
        half: bool = False,
        # training
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = True,
        persistent_workers: bool = False,
        shuffle: bool = False,
        # dataset
        dataset: Optional[Dataset] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str) -> None:
        self.train_dataset = self.hparams["dataset"](
            data_dir=self.hparams["data_dir"],
            split="train",
            chunk_size=self.hparams["chunk_size"],
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
