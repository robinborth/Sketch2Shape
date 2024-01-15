import glob
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

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
        subsample: int = 16384,
        half: bool = False,
    ) -> None:
        self.metainfo = MetaInfo(data_dir=data_dir, split=split)
        self.subsample = subsample
        self.half = half
        self.load()

    def load(self) -> None:
        raise NotImplementedError()

    def fetch(self, idx: int) -> np.ndarray:
        raise NotImplementedError()

    def __len__(self) -> int:
        return self.metainfo.obj_id_count

    def __getitem__(self, idx: int):
        points, gt_sdf = self.fetch(idx)
        return {"points": points, "gt_sdf": gt_sdf, "idx": idx}


class DeepSDFDiskDataset(DeepSDFDatasetBase):
    def fetch(self, idx: int):
        obj_id = self.metainfo.obj_ids[idx]
        points, sdfs = self.metainfo.load_sdf_samples(obj_id=obj_id)
        if self.half:
            points = points.astype(np.float16)
            sdfs = sdfs.astype(np.float16)
        if self.subsample is None:
            return points, sdfs
        random_mask = np.random.choice(points.shape[0], self.subsample)
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
        if self.subsample is None:
            return points, sdfs
        random_mask = np.random.choice(points.shape[0], self.subsample)
        return points[random_mask], sdfs[random_mask]


############################################################
# Evaluation Dataset
############################################################


class LatentOptimizerDataset(Dataset):
    def __init__(self, data_dir: str = "/data"):
        self.data_dir = data_dir
        self.data = []
        for path in glob.glob(self.data_dir + "/*.png"):
            data = {}
            azim, elev, dist = path.split("/")[-1].split("-")[:3]
            camera = Camera(azim=int(azim), elev=-int(elev), dist=int(dist))
            points, rays, mask = camera.unit_sphere_intersection_rays()
            data["points"], data["rays"], data["mask"] = points, rays, mask
            data["camera_position"] = camera.camera_position()
            data["light_position"] = np.array([0, 0, 0], dtype=np.float32)
            data["gt_image"] = plt.imread(path).astype(np.float32)
            data["gt_surface_mask"] = ~np.isclose(data["gt_image"].sum(axis=-1), 3.0)
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


############################################################
# Latent Optimization Datasets
############################################################


class DeepSDFLatentOptimizerDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "/data",
        obj_id: str = "obj_id",
        subsample: int = 16384,
        half: bool = False,
    ):
        self.metainfo = MetaInfo(data_dir=data_dir)
        self.subsample = subsample
        self.label = self.metainfo.load_surface_samples(obj_id=obj_id)
        self.points, self.sdfs = self.metainfo.load_sdf_samples(obj_id=obj_id)
        if half:
            self.points = self.points.astype(np.float16)
            self.sdfs = self.sdfs.astype(np.float16)

    def __len__(self):
        return 1

    def __getitem__(self, idx: int):
        random_mask = np.random.choice(self.points.shape[0], self.subsample)
        return {"gt_points": self.points[random_mask], "label": self.label[random_mask]}


class NormalLatentOptimizerDataset(Dataset):
    # TODO should we use the metainfo here as well?
    def __init__(self, data_dir: str = "/data"):
        self.data_dir = data_dir
        self.data = []
        for path in glob.glob(self.data_dir + "/*.png"):
            data = {}
            azim, elev, dist = path.split("/")[-1].split("-")[:3]
            camera = Camera(azim=int(azim), elev=-int(elev), dist=int(dist))
            points, rays, mask = camera.unit_sphere_intersection_rays()
            data["points"], data["rays"], data["mask"] = points, rays, mask
            data["camera_position"] = camera.camera_position()
            data["light_position"] = np.array([0, 0, 0], dtype=np.float32)
            data["gt_image"] = plt.imread(path).astype(np.float32)
            data["gt_surface_mask"] = ~np.isclose(data["gt_image"].sum(axis=-1), 3.0)
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SketchLatentOptimizerDataset(Dataset):
    pass


############################################################
# Latent Traversal Datasets
############################################################


class LatentTraversalDataset(Dataset):
    pass
