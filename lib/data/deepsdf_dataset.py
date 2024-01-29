import glob
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from lib.data.metainfo import MetaInfo
from lib.render.camera import Camera

############################################################
# DeepSDF Training Datasets
############################################################


class DeepSDFDatasetBase(Dataset):
    def __init__(
        self,
        data_dir: str = "data/",
        split: Optional[str] = None,
        chunk_size: int = 16384,
        half: bool = False,
    ) -> None:
        self.metainfo = MetaInfo(data_dir=data_dir, split=split)
        self.chunk_size = chunk_size
        self.half = half
        self.load()

    def load(self) -> None:
        pass

    def fetch(self, idx: int) -> np.ndarray:
        raise NotImplementedError()

    def __len__(self) -> int:
        return self.metainfo.obj_id_count

    def __getitem__(self, idx: int):
        points, sdf = self.fetch(idx)
        return {"points": points, "sdf": sdf, "idx": idx}


class DeepSDFDiskDataset(DeepSDFDatasetBase):
    def fetch(self, idx: int):
        obj_id = self.metainfo.obj_ids[idx]
        points, sdfs = self.metainfo.load_sdf_samples(obj_id=obj_id)
        if self.half:
            points = points.astype(np.float16)
            sdfs = sdfs.astype(np.float16)
        if self.chunk_size is None:
            return points, sdfs
        random_mask = np.random.choice(points.shape[0], self.chunk_size)
        return points[random_mask], sdfs[random_mask]


class DeepSDFMemoryDataset(DeepSDFDatasetBase):
    def load(self):
        self.points = []
        self.sdfs = []
        for obj_id in self.metainfo.obj_ids:
            points, sdfs = self.metainfo.load_sdf_samples(obj_id=obj_id)
            if self.half:
                points = points.astype(np.float16)
                sdfs = sdfs.astype(np.float16)
            self.points.append(points)
            self.sdfs.append(sdfs)

    def fetch(self, idx: int):
        points, sdfs = self.points[idx], self.sdfs[idx]
        if self.chunk_size is None:
            return points, sdfs
        random_mask = np.random.choice(points.shape[0], self.chunk_size)
        return points[random_mask], sdfs[random_mask]


############################################################
# Evaluation Dataset
############################################################


class SurfaceSamplesDataset(Dataset):
    def __init__(self, data_dir: str = "/data", obj_id: str = "obj_id"):
        self.metainfo = MetaInfo(data_dir=data_dir)
        self.gt_surface_samples = self.metainfo.load_surface_samples(obj_id=obj_id)

    def __len__(self):
        return 1

    def __getitem__(self, idx: int):
        return {"surface_samples": self.gt_surface_samples}


############################################################
# Latent Optimization Datasets
############################################################


class DeepSDFLatentOptimizerDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "/data",
        obj_id: str = "obj_id",
        chunk_size: int = 16384,
        half: bool = False,
    ):
        self.metainfo = MetaInfo(data_dir=data_dir)
        self.chunk_size = chunk_size
        self.points, self.sdfs = self.metainfo.load_sdf_samples(obj_id=obj_id)
        if half:
            self.points = self.points.astype(np.float16)
            self.sdfs = self.sdfs.astype(np.float16)

    def __len__(self):
        return 1

    def __getitem__(self, idx: int):
        random_mask = np.random.choice(self.points.shape[0], self.chunk_size)
        return {"points": self.points[random_mask], "sdf": self.sdfs[random_mask]}


class NormalLatentOptimizerDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "/data",
        obj_id: str = "obj_id",
        azims: list[int] = [],
        elevs: list[int] = [],
        dist: float = 4.0,
    ):
        self.metainfo = MetaInfo(data_dir=data_dir)
        self.transforms = ToTensor()
        self.data = []
        label = 0
        for azim in azims:
            for elev in elevs:
                # HACK to skip views from elev=75
                if label % 8 == 0:
                    label += 1
                    continue
                data = {}
                camera = Camera(azim=azim, elev=elev, dist=dist)
                points, rays, mask = camera.unit_sphere_intersection_rays()
                data["points"], data["rays"], data["mask"] = points, rays, mask
                data["camera_position"] = camera.camera_position()
                normal = self.metainfo.load_normal(obj_id, f"{label:05}")
                data["gt_image"] = self.transforms(normal).permute(1, 2, 0)
                data["gt_surface_mask"] = (data["gt_image"].sum(-1) < 2.95).reshape(-1)
                label += 1
                self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SketchLatentOptimizerDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "/data",
        obj_id: str = "obj_id",
        azims: list[int] = [],
        elevs: list[int] = [],
        dist: float = 4.0,
    ):
        self.metainfo = MetaInfo(data_dir=data_dir)
        self.data = []
        self.transforms = ToTensor()
        label = 0
        for azim in azims:
            for elev in elevs:
                # HACK to skip views from elev=75
                if label % 8 == 0:
                    label += 1
                    continue
                data = {}
                camera = Camera(azim=azim, elev=-elev, dist=dist)
                points, rays, mask = camera.unit_sphere_intersection_rays()
                data["points"], data["rays"], data["mask"] = points, rays, mask
                data["camera_position"] = camera.camera_position()
                sketch = self.metainfo.load_sketch(obj_id, f"{label:05}")
                # sketch = self.metainfo.load_sketch(obj_id, "00019")
                data["sketch"] = self.transforms(sketch).permute(1, 2, 0)
                label += 1
                self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
