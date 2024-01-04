import glob
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset

from lib.data.sdf_utils import remove_nans
from lib.render.camera import Camera


class DeepSDFDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/",
        load_ram: bool = True,
        subsample: int = 16384,
        half: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.load_ram = load_ram
        self.subsample = subsample
        self.half = half
        self._load()

    def _load(self):
        path = Path(self.data_dir)
        self.idx2shape = dict()
        self.npy_paths = list()
        self.data = list()
        for idx, path in enumerate(path.glob("**/*.npz")):
            self.npy_paths.append(str(path))
            self.idx2shape[idx] = path.parts[-1][:-4]
            if self.load_ram:
                self.data.append(self._load_to_ram(path))

        self.shape2idx = {v: k for k, v in self.idx2shape.items()}
        # save shape2idx file
        with open(f"{self.data_dir}/shape2idx.json", "w") as f:
            f.write(json.dumps(self.shape2idx))

    def _load_sdf_samples(self, path):
        data = np.load(path)
        if self.subsample is None:
            return data
        pos_tensor = remove_nans(torch.from_numpy(data["pos"]))
        neg_tensor = remove_nans(torch.from_numpy(data["neg"]))

        # split the sample into half
        half = int(self.subsample / 2)

        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)

        samples = torch.cat([sample_pos, sample_neg], 0)

        return samples

    def _load_sdf_samples_from_ram(self, data):
        if self.subsample is None:
            return data

        hlf = self.subsample // 2

        ### 3 -> 17.198s (with replacement), 25.338 (without replacement)
        # pos_indices = np.random.choice(len(data[0]), hlf, replace=0)
        # neg_indices = np.random.choice(len(data[1]), hlf, replace=hlf > len(data[1]))

        pos_tensor = data[0]
        neg_tensor = data[1]

        pos_size = pos_tensor.shape[0]
        neg_size = neg_tensor.shape[0]

        pos_start_ind = random.randint(0, pos_size - hlf)
        sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + hlf)]

        if neg_size <= hlf:
            random_neg = (torch.rand(hlf) * neg_tensor.shape[0]).long()
            sample_neg = torch.index_select(neg_tensor, 0, random_neg)
        else:
            neg_start_ind = random.randint(0, neg_size - hlf)
            sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + hlf)]

        samples = torch.cat([sample_pos, sample_neg], 0)

        return samples

    def _load_to_ram(self, path):
        data = np.load(path)
        # to make it fit into ram
        pos_tensor = remove_nans(torch.from_numpy(data["pos"])).half()
        neg_tensor = remove_nans(torch.from_numpy(data["neg"])).half()

        return [pos_tensor, neg_tensor]

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        if self.load_ram:
            _data = self._load_sdf_samples_from_ram(self.data[idx])
            idx = np.repeat(idx, len(_data))  # needs to be fixed
            return {
                "xyz": _data[:, :3],
                "sd": _data[:, 3],
                "idx": idx,
            }
        else:
            _data = self._load_sdf_samples(self.npy_paths[idx])
            idx = np.repeat(idx, len(_data))  # needs to be fixed
            return {
                "xyz": _data[:, :3],
                "sd": _data[:, 3],
                "idx": idx,
            }


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
            data["gt_image"] = plt.imread(path).astype(np.float32)
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
