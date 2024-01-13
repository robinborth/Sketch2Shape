import glob
import json
import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset

from lib.data.metainfo import MetaInfo
from lib.data.sdf_utils import remove_nans
from lib.render.camera import Camera


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
        data = self.fetch(idx)
        return {"points": data[:, :3], "gt_sdf": data[:, 3], "idx": idx}


class DeepSDFDataset(DeepSDFDatasetBase):
    def fetch(self, idx: int):
        obj_id = self.metainfo.obj_ids[idx]
        sdf_samples_path = self.metainfo.sdf_samples_path(obj_id=obj_id)
        data = np.load(sdf_samples_path)

        if self.subsample is None:
            return data

        half = self.subsample // 2

        pos_tensor = remove_nans(torch.from_numpy(data["pos"]))
        neg_tensor = remove_nans(torch.from_numpy(data["neg"]))

        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)

        samples = torch.cat([sample_pos, sample_neg], 0)

        return samples


class DeepSDFDatasetMemory(DeepSDFDatasetBase):
    def load(self):
        self.data = []
        for obj_id in self.metainfo.obj_ids:
            sdf_samples_path = self.metainfo.sdf_samples_path(obj_id=obj_id)
            data = np.load(sdf_samples_path)
            pos_tensor = remove_nans(torch.from_numpy(data["pos"])).half()
            neg_tensor = remove_nans(torch.from_numpy(data["neg"])).half()
            self.data.append({"pos": pos_tensor, "neg": neg_tensor})

    def fetch(self, idx: int):
        data = self.data[idx]
        if self.subsample is None:
            return data

        half = self.subsample // 2

        pos_tensor = data["pos"]
        neg_tensor = data["neg"]

        pos_size = pos_tensor.shape[0]
        neg_size = neg_tensor.shape[0]

        pos_start_ind = random.randint(0, pos_size - half)
        sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

        if neg_size <= half:
            random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
            sample_neg = torch.index_select(neg_tensor, 0, random_neg)
        else:
            neg_start_ind = random.randint(0, neg_size - half)
            sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

        samples = torch.cat([sample_pos, sample_neg], 0)

        return samples


class PointCloudDataset(Dataset):
    def __init__(self, ply_path: str, norm_path: str):
        self.ply_path = ply_path
        self.norm_path = norm_path

        self._load()

    def _load(self):
        pointclouds = list()
        shapenet_idxs = list()
        # should match the same files from both folders, as directory is sorted
        # will throw an error if size of the directories is different
        sorted_plyfiles = sorted(list(glob.glob(self.ply_path + "/**/*.ply")))
        sorted_normfiles = sorted(list(glob.glob(self.norm_path + "/**/*.npz")))
        for plyfile, normfile in zip(sorted_plyfiles, sorted_normfiles):
            normfile = np.load(normfile)
            pointcloud = trimesh.load(plyfile)

            # normalize to unit sphere
            pointcloud.vertices = (pointcloud.vertices + normfile["offset"]) * normfile[
                "scale"
            ]

            pointclouds.append(pointcloud)

            shapenet_idx = plyfile.split("/")[-1][:-4]
            shapenet_idxs.append(shapenet_idx)

        self.pointclouds = pointclouds
        self.shapenet_idxs = shapenet_idxs

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        return {
            "shapenet_idx": self.shapenet_idxs[idx],
            "pointcloud": self.pointclouds[idx].vertices,
        }


class RenderedDataset(Dataset):
    def __init__(self, data_dir):
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
