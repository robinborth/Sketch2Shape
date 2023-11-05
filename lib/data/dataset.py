from collections import defaultdict
from pathlib import Path

import cv2
import torchvision.transforms as transforms
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import Dataset

from lib.data.metainfo import MetaInfo


class ShapeNetDatasetBase(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        stage: str = "train",
    ) -> None:
        self.stage = stage
        self.cfg = cfg
        self.metainfo = MetaInfo(cfg=cfg, split=stage)
        self.transform = self._load_transform(cfg=cfg)
        self._load(cfg=cfg)

    def _load_transform(self, cfg: DictConfig):
        trans = []
        if "transform" in cfg:
            trans = [instantiate(trans) for trans in cfg.transform.values()]
        return transforms.Compose(trans)

    def _load(self, cfg: DictConfig):
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


class ShapeNetDatasetDefault(ShapeNetDatasetBase):
    def _load(self, cfg: DictConfig):
        data = defaultdict(lambda: defaultdict(dict))  # type: ignore
        for obj_id in self.metainfo.obj_ids:
            for path in Path(cfg.dataset_path, obj_id, "images").glob("*.jpg"):
                image = cv2.imread(path.as_posix())
                data[obj_id]["images"][path.stem] = self.transform(image)
            for path in Path(cfg.dataset_path, obj_id, "sketches").glob("*.jpg"):
                sketch = cv2.imread(path.as_posix())
                data[obj_id]["sketches"][path.stem] = self.transform(sketch)
        self.data = data

    def _fetch(self, folder: str, obj_id: str, image_id: str):
        return self.data[obj_id][folder][image_id]


class ShapeNetDatasetTransform(ShapeNetDatasetBase):
    def _load(self, cfg: DictConfig):
        data = defaultdict(lambda: defaultdict(dict))  # type: ignore
        for obj_id in self.metainfo.obj_ids:
            for path in Path(cfg.dataset_path, obj_id, "images").glob("*.jpg"):
                image = cv2.imread(path.as_posix())
                data[obj_id]["images"][path.stem] = image
            for path in Path(cfg.dataset_path, obj_id, "sketches").glob("*.jpg"):
                sketch = cv2.imread(path.as_posix())
                data[obj_id]["sketches"][path.stem] = sketch
        self.data = data

    def _fetch(self, folder: str, obj_id: str, image_id: str):
        return self.transform(self.data[obj_id][folder][image_id])


class ShapeNetDatasetFetch(ShapeNetDatasetBase):
    def _fetch(self, folder: str, obj_id: str, image_id: str):
        path = Path(self.cfg.dataset_path, obj_id, f"{folder}/{image_id}.jpg")
        image = cv2.imread(path.as_posix())
        return self.transform(image)
