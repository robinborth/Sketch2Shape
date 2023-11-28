import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DeepSDFDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/",
        load_ram: bool = True,
        subsample: int = 16384,
    ) -> None:
        self.data_dir = data_dir
        self.load_ram = load_ram
        self.subsample = subsample
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
        pos_tensor = self._remove_nans(torch.from_numpy(data["pos"]))
        neg_tensor = self._remove_nans(torch.from_numpy(data["neg"]))

        hlf = self.subsample // 2

        pos_samples = torch.randperm(pos_tensor)[:hlf]
        neg_samples = torch.randperm(neg_tensor)[:hlf]

        samples = torch.cat([pos_samples, neg_samples], dim=0)

        return samples

    def _load_sdf_samples_from_ram(self, data):
        if self.subsample is None:
            return data
        hlf = self.subsample // 2

        pos_indices = torch.randperm(len(data[0]))[:hlf]
        neg_indices = torch.randperm(len(data[1]))[:hlf]

        samples = torch.cat([data[0][pos_indices], data[1][neg_indices]], dim=0)

        return samples

    def _remove_nans(self, tensor):
        tensor_nan = torch.isnan(tensor[:, 3])
        return tensor[~tensor_nan, :]

    def _load_to_ram(self, path):
        data = np.load(path)
        pos_tensor = self._remove_nans(torch.from_numpy(data["pos"]))
        neg_tensor = self._remove_nans(torch.from_numpy(data["neg"]))
        return [pos_tensor, neg_tensor]

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        if self.load_ram:
            _data = self._load_sdf_samples_from_ram(self.data[idx])
            # TODO don't use the idx from the dataloader
            idx = np.repeat(idx, len(_data))  # needs to be fixed
            return {
                "xyz": _data[:, :3],
                "sd": _data[:, 3],
                "idx": idx,
            }
        else:
            _data = self._load_sdf_samples(self.npy_paths[idx])
            idx = np.repeat(idx, len(_data)).reshape(-1, 1)  # needs to be fixed
            return {
                "xyz": _data[:, :3],
                "sd": _data[:, 3],
                "idx": idx,
            }
