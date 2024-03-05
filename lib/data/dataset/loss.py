from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset

from lib.data.metainfo import MetaInfo
from lib.models.deepsdf import DeepSDF

############################################################
# Base Loss Datasets
############################################################


class LossDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/",
        split: Optional[str] = None,
        modes: list[int] = [0, 1],
        sketch_transform: Optional[Callable] = None,
        image_transform: Optional[Callable] = None,
    ):
        self.modes = modes
        self.sketch_transform = sketch_transform
        self.image_transform = image_transform
        self.metainfo = MetaInfo(data_dir=data_dir, split=split)
        self.metainfo.load_loss(modes=modes)

    def __len__(self):
        return self.metainfo.loss_count

    def fetch(self, index):
        info = self.metainfo.get_loss(index)
        image_id = info["image_id"]
        label = info["label"]
        mode = info["mode"]
        image_type = self.metainfo.mode_2_image_type[mode]
        type_idx = self.metainfo.image_type_2_type_idx[image_type]

        image = self.metainfo.load_image(label, image_id, mode)
        if type_idx == 0 and self.sketch_transform:
            image = self.sketch_transform(image)
        if type_idx == 1 and self.image_transform:
            image = self.image_transform(image)

        return {
            "image": image,
            "image_id": int(image_id),
            "type_idx": type_idx,
            "label": label,
        }

    def __getitem__(self, index: int):
        return self.fetch(index)


############################################################
# Latent Loss Dataset
############################################################


class LatentLossDataset(LossDataset):
    def __init__(self, deepsdf_ckpt_path: str = "deepsdf.ckpt", **kwargs):
        super().__init__(**kwargs)
        self.deepsdf = DeepSDF.load_from_checkpoint(
            checkpoint_path=deepsdf_ckpt_path,
            map_location="cpu",
        )
        self.load_latents(modes=self.modes)

    def load_latents(self, modes):
        data = []
        for mode in modes:
            for obj_id in self.metainfo.obj_ids:
                latents = self.metainfo.load_latents(obj_id, mode=mode)
                data.append(latents)
        self._latents = np.concatenate(data)

    def get_latent(self, index: int):
        return self._latents[index]

    def __getitem__(self, index):
        item = self.fetch(index)
        item["latent"] = self.get_latent(index)
        return item
