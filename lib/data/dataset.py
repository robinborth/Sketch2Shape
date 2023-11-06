from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torchvision.transforms as transforms
from omegaconf import DictConfig
from torch.utils.data import Dataset

from lib.data.metainfo import MetaInfo


class ShapeNetDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        stage: str = "train",
    ) -> None:
        self.stage = stage
        self.cfg = cfg
        self.metainfo = MetaInfo(cfg=cfg)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.data = self._load_data(cfg=cfg)

    def _load_data(self, cfg: DictConfig):
        data = defaultdict(lambda: defaultdict(dict))  # type: ignore
        for obj_id in cfg.obj_ids:
            for path in Path(cfg.dataset_path, obj_id, "images").glob("*.jpg"):
                image = cv2.imread(path.as_posix())
                data[obj_id]["images"][path.stem] = self.transform(image)
            for path in Path(cfg.dataset_path, obj_id, "sketches").glob("*.jpg"):
                sketch = cv2.imread(path.as_posix())
                data[obj_id]["sketches"][path.stem] = self.transform(sketch)
        return data

    def __len__(self):
        return len(self.metainfo)

    def __getitem__(self, index):
        info = self.metainfo[index]
        obj_id = info["obj_id"]
        image_id = info["image_id"]
        sketch_id = info["sketch_id"]
        label = info["label"]

        sketch = self.data[obj_id]["sketches"][sketch_id]
        image = self.data[obj_id]["images"][image_id]

        return {
            "sketch": sketch,
            "sketch_id": image_id,
            "image": image,
            "image_id": image_id,
            "label": label,
        }


class SDFDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        self.cfg = cfg
        self._load_datapaths(cfg=cfg)

    def _load_datapaths(self, cfg: DictConfig):
        p = Path(cfg.data_path)
        paths = list()
        shape2idx = dict()
        data = list()

        for idx, file in enumerate(p.glob("**/*.npy")):
            paths.append(file)
            shape2idx[file] = idx
            samples = load_samples(file)
            # append shape_idx information
            samples = np.column_stack((samples, np.repeat(idx, samples.shape[0])))
            data.append(samples)

        data = np.stack(data).reshape(-1, 5)
        self.npy_paths = paths
        self.shape2idx = shape2idx
        self.idx2shape = {v: k for k, v in shape2idx.items()}
        # data: [x, y, z, sdf, shape_idx]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # self.data: [500_000, 5]
        # collate_fn
        data = self.data[idx].copy()
        xyz = data[:3]
        sd = data[3]
        key = data[4].astype(np.int32)
        return {"xyz": xyz, "sd": sd, "key": key}


def load_samples(filename, subsample=None):
    npz = np.load(filename)
    return npz
