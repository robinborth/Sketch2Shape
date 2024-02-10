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
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.obj_id = obj_id

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = self.hparams["train_dataset"](obj_id=self.obj_id)
        if stage == "test":
            self.eval_dataset = self.hparams["eval_dataset"](obj_id=self.obj_id)

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

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.eval_dataset,
            batch_size=1,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            persistent_workers=self.hparams["persistent_workers"],
        )
