from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
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

    def load_sketch_image_pairs(self):
        data = []
        for _, row in self._metainfo.iterrows():
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

    def load_mesh(
        self,
        obj_id: str,
        file_name: str = "normalized_mesh.obj",
        normalize: bool = False,
    ) -> o3d.geometry.TriangleMesh:
        path = self.data_dir / "shapes" / obj_id / file_name
        mesh = o3d.io.read_triangle_mesh(str(path))
        if not normalize:
            return mesh

        points = np.asarray(mesh.vertices)
        translate = (np.min(points, axis=0) + np.max(points, axis=0)) / 2.0
        points -= translate
        points /= np.linalg.norm(points, axis=-1).max()
        mesh.vertices = o3d.utility.Vector3dVector(points)
        return mesh

    def save_mesh(
        self,
        obj_id: str,
        mesh: o3d.geometry.TriangleMesh,
    ) -> None:
        path = self.data_dir / "shapes" / obj_id / "mesh.obj"
        o3d.io.write_triangle_mesh(str(path), mesh)

    def save_normalized_mesh(
        self,
        obj_id: str,
        mesh: o3d.geometry.TriangleMesh,
    ) -> None:
        path = self.data_dir / "shapes" / obj_id / "normalized_mesh.obj"
        o3d.io.write_triangle_mesh(str(path), mesh)

    def load_sdf_samples(self, obj_id: str) -> Tuple[np.ndarray, np.ndarray]:
        path = self.data_dir / "shapes" / obj_id / "sdf_samples.npz"
        data = np.load(path)
        points = data[:, :3]
        sdfs = data[:, 3]
        return points, sdfs

    def save_sdf_samples(self, obj_id: str, points: np.ndarray, sdfs: np.ndarray):
        path = self.data_dir / "shapes" / obj_id / "sdf_samples.npz"
        data = np.concatenate([points, sdfs], axis=-1)
        np.save(path, data)

    def load_surface_samples(self, obj_id: str) -> np.ndarray:
        path = self.data_dir / "shapes" / obj_id / "surface_samples.npz"
        return np.load(path)

    def save_surface_samples(self, obj_id: str, points: np.ndarray, ) -> None:
        path = self.data_dir / "shapes" / obj_id / "surface_samples.npz"
        return np.save(path, points)

    def render_path(self, obj_id: str, render_type: str, image_id: str) -> Path:
        return self.data_dir / "shapes" / obj_id / render_type / f"{image_id}.jpg"
