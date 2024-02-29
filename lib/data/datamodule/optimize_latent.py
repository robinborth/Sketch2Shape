from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class LatentOptimizationDataModule(LightningDataModule):
    def __init__(
        self,
        obj_id: str = "obj_id",
        # training
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = True,
        persistent_workers: bool = False,
        shuffle: bool = False,
        # dataset
        dataset: Optional[Dataset] = None,
        milestones: list[int] = [],
        size: int = 256,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.obj_id = obj_id
        self.size = size
        self.milestones = milestones

    def get_sizes(self):
        return [self.size // (2**n) for n in range(len(self.milestones) + 1)]

    def get_dataset(self):
        current_epoch = self.trainer.current_epoch
        num_downsample = sum(m >= current_epoch for m in self.milestones)
        return self.datasets[num_downsample]

    def setup(self, stage: str) -> None:
        self.datasets = []
        for size in self.get_sizes():
            dataset = self.hparams["dataset"](obj_id=self.obj_id, size=size)
            self.datasets.append(dataset)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.get_dataset(),
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            shuffle=self.hparams["shuffle"],
        )
