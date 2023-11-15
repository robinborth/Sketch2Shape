from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import cv2
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from lib.data.metainfo import MetaInfo


class SiameseDatasetBase(Dataset):
    def __init__(
        self,
        data_dir: str = "data/",
        stage: str = "train",
        transforms: Optional[Callable] = None,
    ):
        self.data_dir = data_dir
        self.transforms = transforms if transforms else Compose()
        self.metainfo = MetaInfo(data_dir=data_dir, split=stage)
        self._load()

    def _load(self):
        pass

    def _fetch(self, folder: str, obj_id: str, image_id: str):
        pass

    def __len__(self):
        return self.metainfo.pair_count

    def __getitem__(self, index):
        info = self.metainfo.get_pair(index)
        obj_id = info["obj_id"]
        image_id = info["image_id"]
        sketch_id = info["sketch_id"]
        label = info["label"]

        sketch = self._fetch("sketches", obj_id, sketch_id)
        image = self._fetch("images", obj_id, image_id)

        return {
            "sketch": sketch,
            "image": image,
            "label": label,
            "image_id": image_id,
            "sketch_id": image_id,
        }


class SiameseDatasetPreLoadPreTransform(SiameseDatasetBase):
    def _load(self):
        data = defaultdict(lambda: defaultdict(dict))  # type: ignore
        for obj_id in self.metainfo.obj_ids:
            for path in Path(self.data_dir, obj_id, "images").glob("*.jpg"):
                image = cv2.imread(path.as_posix())
                data[obj_id]["images"][path.stem] = self.transforms(image)
            for path in Path(self.data_dir, obj_id, "sketches").glob("*.jpg"):
                sketch = cv2.imread(path.as_posix())
                data[obj_id]["sketches"][path.stem] = self.transforms(sketch)
        self.data = data

    def _fetch(self, folder: str, obj_id: str, image_id: str):
        return self.data[obj_id][folder][image_id]


class SiameseDatasetPreLoadDynamicTransform(SiameseDatasetBase):
    def _load(self):
        data = defaultdict(lambda: defaultdict(dict))  # type: ignore
        for obj_id in self.metainfo.obj_ids:
            for path in Path(self.data_dir, obj_id, "images").glob("*.jpg"):
                image = cv2.imread(path.as_posix())
                data[obj_id]["images"][path.stem] = image
            for path in Path(self.data_dir, obj_id, "sketches").glob("*.jpg"):
                sketch = cv2.imread(path.as_posix())
                data[obj_id]["sketches"][path.stem] = sketch
        self.data = data

    def _fetch(self, folder: str, obj_id: str, image_id: str):
        return self.transforms(self.data[obj_id][folder][image_id])


class SiameseDatasetDynamicLoadDynamicTransform(SiameseDatasetBase):
    def _fetch(self, folder: str, obj_id: str, image_id: str):
        path = Path(self.data_dir, obj_id, f"{folder}/{image_id}.jpg")
        image = cv2.imread(path.as_posix())
        return self.transforms(image)
