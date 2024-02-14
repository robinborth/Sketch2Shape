from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from lib.data.metainfo import MetaInfo


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
