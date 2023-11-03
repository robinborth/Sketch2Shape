import numpy as np
import pandas as pd
from omegaconf import DictConfig


class MetaInfo:
    def __init__(self, cfg: DictConfig):
        dtype = {"image_id": str, "sketch_id": str}
        self.df = pd.read_csv(cfg.metainfo_path, dtype=dtype)

    @property
    def labels(self):
        return np.array(self.df["label"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        return self.df.iloc[index].to_dict()
