import numpy as np
from torch.utils.data import Dataset

from lib.data.metainfo import MetaInfo
from lib.data.transforms import BaseTransform
from lib.render.camera import Camera

############################################################
# DeepSDF Optimization Datasets
############################################################


class DeepSDFLatentOptimizerDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "/data",
        obj_id: str = "obj_id",
        chunk_size: int = 16384,
        half: bool = False,
        **kwargs,
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


############################################################
# Normal Optimization Datasets
############################################################


class NormalLatentOptimizerDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "/data",
        obj_id: str = "obj_id",
        azims: list[int] = [],
        elevs: list[int] = [],
        dist: float = 4.0,
        size: int = 256,
        **kwargs,
    ):
        self.metainfo = MetaInfo(data_dir=data_dir)
        self.transforms = BaseTransform(
            to_image=False,
            normalize=False,
            size=size,
            sharpness=1.0,
        )
        self.data = []
        label = 0
        for azim in azims:
            for elev in elevs:
                data = {}
                camera = Camera(
                    azim=azim,
                    elev=elev,
                    dist=dist,
                    height=size,
                    width=size,
                    focal=size * 2,
                )
                points, rays, mask = camera.unit_sphere_intersection_rays()
                data["points"], data["rays"], data["mask"] = points, rays, mask
                data["camera_position"] = camera.camera_position()
                normal = self.metainfo.load_normal(obj_id, f"{label:05}")
                data["gt_image"] = self.transforms(normal)  # (H, W, 3)
                data["gt_surface_mask"] = (data["gt_image"].sum(-1) < 2.95).reshape(-1)
                label += 1
                self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


############################################################
# Sketch Optimization Datasets
############################################################


class SketchLatentOptimizerDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "/data",
        obj_id: str = "obj_id",
        azims: list[int] = [],
        elevs: list[int] = [],
        dist: float = 4.0,
        sketch_id: int = 11,
        size: int = 256,
        **kwargs,
    ):
        self.metainfo = MetaInfo(data_dir=data_dir)
        self.data = []
        self.transforms = BaseTransform(
            to_image=True,
            normalize=True,
            size=size,
            sharpness=1.0,
        )
        label = 0
        for azim in azims:
            for elev in elevs:
                data = {}
                camera = Camera(
                    azim=azim,
                    elev=elev,
                    dist=dist,
                    height=size,
                    width=size,
                    focal=size * 2,
                )
                points, rays, mask = camera.unit_sphere_intersection_rays()
                data["points"], data["rays"], data["mask"] = points, rays, mask
                data["camera_position"] = camera.camera_position()
                sketch = self.metainfo.load_sketch(obj_id, f"{sketch_id:05}")
                data["sketch"] = self.transforms(sketch)  # (3, W, H)
                label += 1
                self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
