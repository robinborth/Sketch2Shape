from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset

from lib.data.metainfo import MetaInfo


class SiameseDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/",
        split: Optional[str] = None,
        sketch_transforms: Optional[Callable] = None,
        image_transforms: Optional[Callable] = None,
    ):
        self.sketch_transforms = sketch_transforms
        self.image_transforms = image_transforms
        self.metainfo = MetaInfo(data_dir=data_dir, split=split)
        self.metainfo.load_snn()

    def __len__(self):
        return self.metainfo.snn_count

    def __getitem__(self, index):
        info = self.metainfo.get_snn(index)
        obj_id = info["obj_id"]
        image_id = info["image_id"]
        label = info["label"]
        image_type = info["image_type"]

        if image_type == "sketch":
            image = self.metainfo.load_sketch(obj_id, image_id)
            if self.sketch_transforms is not None:
                image = self.sketch_transforms(image)

        if image_type == "normal":
            image = self.metainfo.load_normal(obj_id, image_id)
            if self.image_transforms is not None:
                image = self.image_transforms(image)

        return {
            "image": image,
            "image_id": int(image_id),
            "image_type": self.metainfo.image_type_2_type_idx[image_type],
            "label": label,
        }


class SiameseBatchDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/",
        split: Optional[str] = None,
        sketch_transforms: Optional[Callable] = None,
        image_transforms: Optional[Callable] = None,
    ):
        self.sketch_transforms = sketch_transforms
        self.image_transforms = image_transforms
        self.metainfo = MetaInfo(data_dir=data_dir, split=split)
        self.metainfo.load_snn()
        self.labels = np.unique(self.metainfo.snn_labels, return_inverse=True)[1]
        _, counts = np.unique(self.labels, return_counts=True)
        self.num_images = counts[0]
        assert np.all(counts == self.num_images)

    def __len__(self):
        return self.metainfo.obj_id_count

    def __getitem__(self, idx: int):
        data = []
        for image_idx in range(self.num_images):
            snn_idx = (idx * self.num_images) + image_idx
            info = self.metainfo.get_snn(snn_idx)
            obj_id = info["obj_id"]
            image_id = info["image_id"]
            label = info["label"]
            image_type = info["image_type"]

            if image_type == "sketch":
                image = self.metainfo.load_sketch(obj_id, image_id)
                if self.sketch_transforms is not None:
                    image = self.sketch_transforms(image)

            if image_type == "normal":
                image = self.metainfo.load_normal(obj_id, image_id)
                if self.image_transforms is not None:
                    image = self.image_transforms(image)

            data.append(
                {
                    "image": image,
                    "image_id": int(image_id),
                    "image_type": self.metainfo.image_type_2_type_idx[image_type],
                    "label": label,
                }
            )

        return data
