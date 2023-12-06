from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from lib.data.metainfo import MetaInfo


class SiameseDatasetBase(Dataset):
    def __init__(
        self,
        metainfo: MetaInfo,
        transforms: Optional[Callable] = None,
    ):
        self.transforms = transforms if transforms else Compose()
        self.metainfo = metainfo
        self.data_dir = metainfo.data_dir
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
        label = info["label"]

        sketch = self._fetch("sketches", obj_id, image_id)
        image = self._fetch("images", obj_id, image_id)

        return {
            "sketch": sketch,
            "image": image,
            "label": label,
            "image_id": image_id,
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


class SiameseChunkDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/",
        stage: str = "train",
        transforms: Optional[Callable] = None,
    ):
        self.data_dir = data_dir
        self.transforms = transforms if transforms else Compose()
        self.metainfo = MetaInfo(data_dir=data_dir, split=stage)

    def _fetch(self, folder: str, obj_id: str):
        paths = Path(self.data_dir, obj_id, folder).glob("*.jpg")
        images = []
        for path in paths:
            image = cv2.imread(path.as_posix())
            images.append(self.transforms(image))
        return np.stack(images)

    def __len__(self):
        return self.metainfo.obj_id_count

    def __getitem__(self, index):
        obj_id = self.metainfo.obj_ids[index]
        sketch = self._fetch("sketches", obj_id)
        image = self._fetch("images", obj_id)
        label = np.repeat(index, len(sketch))
        return {
            "sketch": sketch,
            "image": image,
            "label": label,
        }


class SiameseH5pyDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/",
        stage: str = "train",
        transforms: Optional[Callable] = None,
    ):
        self.data_dir = data_dir
        self.transforms = transforms if transforms else Compose()
        self.metainfo = MetaInfo(data_dir=data_dir, split=stage)

    def _fetch(self, folder: str, obj_id: str):
        hd5py_file_name = f"{Path(self.data_dir).stem}.h5"
        h5_file = h5py.File(Path(self.data_dir, obj_id, hd5py_file_name), "r+")
        data = np.array(h5_file[folder].astype(np.uint8))
        h5_file.close()
        return np.stack([self.transforms(img) for img in data])

    def __len__(self):
        return self.metainfo.obj_id_count

    def __getitem__(self, index):
        obj_id = self.metainfo.obj_ids[index]
        sketch = self._fetch("sketches", obj_id)
        image = self._fetch("images", obj_id)
        label = np.repeat(index, len(sketch))
        return {
            "sketch": sketch,
            "image": image,
            "label": label,
        }


class SiameseDatasetEasyImages(SiameseDatasetBase):
    def _fetch(self, folder: str, obj_id: str, image_id: str):
        paths = [
            Path(self.data_dir, obj_id, f"{folder}/00014.jpg"),
            Path(self.data_dir, obj_id, f"{folder}/00015.jpg"),
            Path(self.data_dir, obj_id, f"{folder}/00022.jpg"),
            Path(self.data_dir, obj_id, f"{folder}/00023.jpg"),
        ]
        images = []
        for path in paths:
            image = cv2.imread(path.as_posix())
            images.append(self.transforms(image))
        return np.stack(images)

    def __len__(self):
        return self.metainfo.obj_id_count

    def __getitem__(self, index):
        obj_id = self.metainfo.obj_ids[index]
        sketch = self._fetch("sketches", obj_id, "")
        image = self._fetch("images", obj_id, "")
        label = np.repeat(index, len(sketch))
        return {
            "obj_id": obj_id,
            "sketch": sketch,
            "image": image,
            "label": label,
        }
