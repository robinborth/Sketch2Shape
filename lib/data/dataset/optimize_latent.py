import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from lib.data.metainfo import MetaInfo
from lib.data.transforms import BaseTransform, SketchTransform
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
        self.transforms = BaseTransform(transforms=[v2.Resize((size, size))])
        self.data = []
        view_id = 0
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
                label = self.metainfo.obj_id_to_label(obj_id)
                normal = self.metainfo.load_image(label, view_id, 1)
                data["gt_image"] = self.transforms(normal)  # (H, W, 3)
                data["gt_surface_mask"] = (data["gt_image"].sum(-1) < 2.95).reshape(-1)
                view_id += 1
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
        view_id: int = 6,  # 0
        mode: int = 9,  # 9
        size: int = 256,
        **kwargs,
    ):
        self.metainfo = MetaInfo(data_dir=data_dir)
        self.data = []
        self.transforms = SketchTransform()
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
                data["world_to_camera"] = camera.get_world_to_camera()
                data["camera_width"] = size
                data["camera_height"] = size
                data["camera_focal"] = size * 2
                label = self.metainfo.obj_id_to_label(obj_id)
                sketch = self.metainfo.load_image(label, view_id, mode)
                data["sketch"] = self.transforms(sketch)  # (3, W, H)
                self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


############################################################
# Inference Optimization Datasets
############################################################


class InferenceOptimizerDataset(Dataset):
    def __init__(
        self,
        sketch: Image,
        silhouettes: list = [],
        azims: list[int] = [],
        elevs: list[int] = [],
        dist: float = 4.0,
        size: int = 256,
        **kwargs,
    ):
        self.data = []
        self.transforms = SketchTransform()
        for azim, elev, silhouette in zip(azims, elevs, silhouettes):
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
            data["world_to_camera"] = camera.get_world_to_camera()
            data["camera_width"] = size
            data["camera_height"] = size
            data["camera_focal"] = size * 2
            data["sketch"] = self.transforms(sketch)  # (3, H, W)
            silhouette = np.array(silhouette).sum(-1) < 600
            data["silhouette"] = silhouette.astype(np.float32)
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
