from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
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
        self._load(cfg=cfg)

    def _load(self, cfg: DictConfig):
        path = Path(cfg.data_path)
        self.idx2shape = dict()
        self.npy_paths = list()
        self.data = list()
        for idx, path in enumerate(path.glob("**/*npz")):
            self.npy_paths.append(str(path))
            self.idx2shape[idx] = path.parts[-3] + "/" + path.parts[-2]
            if self.cfg.load_ram:
                self.data.append(load_to_ram(path))

        self.shape2idx = {v: k for k, v in self.idx2shape.items()}

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        if self.cfg.load_ram:
            _data = load_sdf_samples_from_ram(self.data[idx], self.cfg.subsample)
            idx = np.repeat(idx, len(_data))  # needs to be fixed
            return {
                "xyz": _data[:, :3],
                "sd": _data[:, 3],
                "idx": idx,
            }
        else:
            _data = load_sdf_samples(self.npy_paths[idx], self.cfg.subsample)
            idx = np.repeat(idx, len(_data)).reshape(-1, 1)  # needs to be fixed
            return {
                "xyz": _data[:, :3],
                "sd": _data[:, 3],
                "idx": idx,
            }


def load_sdf_samples(path, subsample=None):
    data = np.load(path)
    if subsample is None:
        return data
    pos_tensor = torch.from_numpy(data["pos"])
    neg_tensor = torch.from_numpy(data["neg"])

    hlf = subsample // 2

    pos_samples = torch.randperm(pos_tensor)[:hlf]
    neg_samples = torch.randperm(neg_tensor)[:hlf]

    samples = torch.cat([pos_samples, neg_samples], dim=0)

    return samples


def load_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    hlf = subsample // 2

    pos_indices = torch.randperm(len(data[0]))[:hlf]
    neg_indices = torch.randperm(len(data[1]))[:hlf]

    samples = torch.cat([data[0][pos_indices], data[1][neg_indices]], dim=0)

    return samples


def store_dicts(dicts: list[dict]):
    pass


def load_to_ram(path):
    data = np.load(path)
    pos_tensor = torch.from_numpy(data["pos"])
    neg_tensor = torch.from_numpy(data["neg"])

    return [pos_tensor, neg_tensor]


# -------- for overfitting experiments
class SDFDummySimple(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        self.cfg = cfg
        self._load(cfg=cfg)

    def _load(self, cfg: DictConfig):
        self.data = np.array([[-1, -1, -1, -0.8], [1, 1, 1, 0.8]], dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx].copy()
        return {
            "xyz": data[:3],
            "sd": data[3],
        }


class SDFDummyMedium(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        self.cfg = cfg
        self._load(cfg=cfg)

    def _load(self, cfg: DictConfig):
        path = Path(cfg.data_path)
        data = [np.load(obj_file) for obj_file in path.glob("**/*.npy")]
        self.data = np.stack(data).reshape(-1, 4)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # data = self.data[idx].copy()
        return {
            "xyz": self.data[:, :3],
            "sd": self.data[:, 3],
        }
