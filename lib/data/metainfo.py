from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from lib.utils import create_logger

logger = create_logger("metainfo")


class MetaInfo:
    def __init__(self, data_dir: str = "data/", split: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.metainfo_path = self.data_dir / "metainfo.csv"
        try:
            metainfo = pd.read_csv(self.metainfo_path)
            if split is not None:
                metainfo = metainfo[metainfo["split"] == split]
            self._obj_ids = metainfo["obj_id"].to_list()
            self._metainfo = metainfo
        except Exception as e:
            logger.error("Not able to load dataset_splits file.")

        data = []
        for _, row in metainfo.iterrows():
            obj_id, label = row["obj_id"], row["label"]
            images_path = self.data_dir / "shapes" / obj_id / "images"
            assert images_path.exists()
            for image_file in sorted(images_path.iterdir()):
                image_id = image_file.stem
                data.append(dict(obj_id=obj_id, image_id=image_id, label=label))
        self._sketch_image_pairs = pd.DataFrame(data)

    @property
    def labels(self):
        return np.array(self._sketch_image_pairs["label"])

    @property
    def pair_count(self):
        return len(self._sketch_image_pairs)

    def get_pair(self, index: int):
        return self._sketch_image_pairs.iloc[index].to_dict()

    @property
    def obj_ids(self):
        return self._obj_ids

    @property
    def obj_id_count(self):
        return len(self.obj_ids)

    def label_to_obj_id(self, label: int) -> str:
        df = self._metainfo
        return df.loc[df["label"] == str(label)].iloc[0]["obj_id"]

    def obj_id_to_label(self, obj_id: str) -> int:
        df = self._metainfo
        return int(df.loc[df["obj_id"] == obj_id].iloc[0]["label"])

    def model_normalized_path(self, obj_id: str) -> Path:
        return self.data_dir / "shapes" / obj_id / "model_normalized.obj"

    def normalization_path(self, obj_id: str) -> Path:
        return self.data_dir / "shapes" / obj_id / "normalization.npz"

    def sdf_samples_path(self, obj_id: str) -> Path:
        return self.data_dir / "shapes" / obj_id / "sdf_samples.npz"

    def surface_samples_path(self, obj_id: str) -> Path:
        return self.data_dir / "shapes" / obj_id / "surface_samples.ply"

    def render_path(self, obj_id: str, render_type: str, image_id: str) -> Path:
        return self.data_dir / "shapes" / obj_id / render_type / f"{image_id}.jpg"
