from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset

from lib.data.metainfo import MetaInfo


class LossTesterDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/",
        split: Optional[str] = None,
        modes: list[int] = [0, 1],
        sketch_transform: Optional[Callable] = None,
        normal_transform: Optional[Callable] = None,
    ):
        self.sketch_transform = sketch_transform
        self.normal_transform = normal_transform
        self.metainfo = MetaInfo(data_dir=data_dir, split=split)
        self.metainfo.load_loss(modes=modes)
        self.labels = np.unique(self.metainfo.loss_labels, return_inverse=True)[1]
        _, counts = np.unique(self.labels, return_counts=True)
        self.num_images = counts[0]
        assert np.all(counts == self.num_images)
        self.sorted_idx = np.argsort(self.labels)

    def __len__(self):
        return self.metainfo.obj_id_count

    def __getitem__(self, idx: int):
        data = []
        for image_idx in range(self.num_images):
            loss_idx = (idx * self.num_images) + image_idx
            loss_idx = self.sorted_idx[loss_idx]
            info = self.metainfo.get_loss(loss_idx)
            image_id = info["image_id"]
            label = info["label"]
            mode = info["mode"]
            image_type = self.metainfo.mode_2_image_type[mode]
            type_idx = self.metainfo.image_type_2_type_idx[image_type]

            image = self.metainfo.load_image(label, image_id, mode)
            if type_idx == 0 and self.sketch_transform:
                image = self.sketch_transform(image)
            if type_idx == 1 and self.normal_transform:
                image = self.normal_transform(image)

            data.append(
                {
                    "image": image,
                    "image_id": int(image_id),
                    "type_idx": type_idx,
                    "label": label,
                }
            )

        return data
