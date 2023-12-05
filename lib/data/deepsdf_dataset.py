import json
import random
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

        ### 3 -> 17.198s (with replacement), 25.338 (without replacement)
        # pos_indices = np.random.choice(len(data[0]), hlf, replace=0)
        # neg_indices = np.random.choice(len(data[1]), hlf, replace=0)

        ### 4 -> 15.653 (official implementation)
        pos_tensor = data[0]
        neg_tensor = data[1]

        # split the sample into half
        half = int(self.subsample / 2)

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

    def _remove_nans(self, tensor):
        tensor_nan = torch.isnan(tensor[:, 3])
        return tensor[~tensor_nan, :]

    def _load_to_ram(self, path):
        data = np.load(path)
        # to make it fit into ram
        pos_tensor = self._remove_nans(torch.from_numpy(data["pos"]))
        neg_tensor = self._remove_nans(torch.from_numpy(data["neg"]))

        ### 3 -> 17.198s (with replacement), 25.338 (without replacement)
        # pos_indices = np.random.choice(len(data[0]), hlf, replace=0)
        # neg_indices = np.random.choice(len(data[1]), hlf, replace=0)

        pos_shuffle = torch.randperm(pos_tensor.shape[0])
        neg_shuffle = torch.randperm(neg_tensor.shape[0])

        return [pos_tensor[pos_shuffle], neg_tensor[neg_shuffle]]

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
            idx = np.repeat(idx, len(_data))  # needs to be fixed
            return {
                "xyz": _data[:, :3],
                "sd": _data[:, 3],
                "idx": idx,
            }
