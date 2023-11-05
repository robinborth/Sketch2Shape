from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from lib.utils import create_logger

logger = create_logger("metainfo")


class MetaInfo:
    def __init__(self, cfg: DictConfig, split: Optional[str] = None):
        self.split = split
        self.cfg = cfg

        try:
            dataset_splits = pd.read_csv(cfg.dataset_splits_path)
            if split is not None:
                dataset_splits = dataset_splits[dataset_splits["split"] == split]
            self._obj_ids = dataset_splits["obj_id"].to_list()
            self._dataset_splits = dataset_splits
        except Exception as e:
            logger.error("Not able to load dataset_splits file.")

        # load the sketch_image pairs file
        dtype = {"image_id": str, "sketch_id": str}
        try:
            pairs = pd.read_csv(cfg.sketch_image_pairs_path, dtype=dtype)
            if split is not None:
                pairs = pairs[pairs["split"] == split]
            self._sketch_image_pairs = pairs
        except Exception as e:
            logger.error("Not able to load sketch_image_pairs file.")

    @property
    def labels(self):
        return np.array(self._sketch_image_pairs["label"])

    @property
    def obj_ids(self):
        return self._obj_ids

    @property
    def obj_ids_splits(self):
        for row in self._dataset_splits.itertuples():
            yield row.obj_id, row.split

    @property
    def pair_count(self):
        return len(self._sketch_image_pairs)

    @property
    def obj_id_count(self):
        return len(self.obj_ids)

    def get_pair(self, index: int):
        return self._sketch_image_pairs.iloc[index].to_dict()
