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
            obj_id = info["obj_id"]
            image_id = info["image_id"]
            label = info["label"]
            image_type = self.metainfo.mode_2_image_type[info["mode"]]
            type_idx = self.metainfo.mode_2_type_idx[info["mode"]]

            if image_type == "sketch":
                image = self.metainfo.load_sketch(obj_id, image_id)
                if self.sketch_transform is not None:
                    image = self.sketch_transform(image)

            if image_type == "rendered_sketch":
                image = self.metainfo.load_rendered_sketch(obj_id, image_id)
                if self.sketch_transform is not None:
                    image = self.sketch_transform(image)

            if image_type == "normal":
                image = self.metainfo.load_normal(obj_id, image_id)
                if self.normal_transform is not None:
                    image = self.normal_transform(image)

            if image_type == "rendered_normal":
                image = self.metainfo.load_rendered_normal(obj_id, image_id)
                if self.normal_transform is not None:
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
