import shutil
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from PIL import Image

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

    #################################################################
    # SNN pairs loader utils
    #################################################################

    def load_sketch_image_pairs(self):
        data = []
        for _, row in self._metainfo.iterrows():
            obj_id, label = row["obj_id"], row["label"]
            images_path = self.data_dir / "shapes" / obj_id / "sketches"
            assert images_path.exists()
            for image_file in sorted(images_path.iterdir()):
                image_id = image_file.stem
                data.append(dict(obj_id=obj_id, image_id=image_id, label=label))
        self._sketch_image_pairs = pd.DataFrame(data)

    @property
    def pair_count(self):
        return len(self._sketch_image_pairs)

    def get_pair(self, index: int):
        return self._sketch_image_pairs.iloc[index].to_dict()

    #################################################################
    # Object IDs and Labels
    #################################################################

    @property
    def obj_ids(self):
        return self._obj_ids

    @property
    def obj_id_count(self):
        return len(self.obj_ids)

    @property
    def labels(self):
        return np.array(self._sketch_image_pairs["label"])

    def label_to_obj_id(self, label: int) -> str:
        df = self._metainfo
        return df.loc[df["label"] == str(label)].iloc[0]["obj_id"]

    def obj_id_to_label(self, obj_id: str) -> int:
        df = self._metainfo
        filtered = df.loc[df["obj_id"] == obj_id]
        if not len(filtered):
            return -1
        return int(filtered.iloc[0]["label"])

    #################################################################
    # Mesh: Loading and Storing Utils
    #################################################################

    def mesh_path(self, obj_id: str) -> Path:
        return self.data_dir / "shapes" / obj_id / "mesh.obj"

    def load_mesh(
        self, obj_id: str, normalize: bool = False
    ) -> o3d.geometry.TriangleMesh:
        path = self.mesh_path(obj_id)
        mesh = o3d.io.read_triangle_mesh(str(path))
        if not normalize:
            return mesh

        points = np.asarray(mesh.vertices)
        translate = (np.min(points, axis=0) + np.max(points, axis=0)) / 2.0
        points -= translate
        points /= np.linalg.norm(points, axis=-1).max()
        mesh.vertices = o3d.utility.Vector3dVector(points)
        return mesh

    def save_mesh(self, source_path: Path, obj_id: str) -> None:
        path = self.mesh_path(obj_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_path, path)

    #################################################################
    # Normalized Mesh: Loading and Storing Utils
    #################################################################

    def normalized_mesh_path(self, obj_id: str) -> Path:
        return self.data_dir / "shapes" / obj_id / "normalized_mesh.obj"

    def load_normalized_mesh(self, obj_id: str) -> o3d.geometry.TriangleMesh:
        path = self.normalized_mesh_path(obj_id)
        return o3d.io.read_triangle_mesh(str(path))

    def save_normalized_mesh(
        self, obj_id: str, mesh: o3d.geometry.TriangleMesh
    ) -> None:
        path = self.normalized_mesh_path(obj_id)
        o3d.io.write_triangle_mesh(str(path), mesh=mesh, write_triangle_uvs=False)

    #################################################################
    # SDF Samples: Loading and Storing Utils
    #################################################################

    def sdf_samples_path(self, obj_id: str) -> Path:
        return self.data_dir / "shapes" / obj_id / "sdf_samples.npy"

    def load_sdf_samples(self, obj_id: str) -> Tuple[np.ndarray, np.ndarray]:
        path = self.sdf_samples_path(obj_id)
        data = np.load(path).astype(np.float32)
        points = data[:, :3]
        sdfs = data[:, 3]
        return points, sdfs

    def save_sdf_samples(self, obj_id: str, samples: np.ndarray):
        path = self.sdf_samples_path(obj_id)
        np.save(path, samples.astype(np.float32))

    #################################################################
    # Surface Samples: Loading and Storing Utils
    #################################################################

    def surface_samples_path(self, obj_id: str) -> Path:
        return self.data_dir / "shapes" / obj_id / "surface_samples.npy"

    def load_surface_samples(self, obj_id: str) -> np.ndarray:
        path = self.surface_samples_path(obj_id)
        return np.load(path).astype(np.float32)

    def save_surface_samples(self, obj_id: str, samples: np.ndarray) -> None:
        path = self.surface_samples_path(obj_id)
        return np.save(path, samples.astype(np.float32))

    #################################################################
    # Normals: Loading and Storing Utils
    #################################################################

    def normals_dir_path(self, obj_id: str) -> Path:
        return self.data_dir / "shapes" / obj_id / "normals"

    def save_normal(self, normals: np.ndarray, obj_id: str, image_id: str):
        path = self.normals_dir_path(obj_id) / f"{image_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(normals).save(path)

    def load_normal(self, obj_id: str, image_id: str) -> Path:
        path = self.normals_dir_path(obj_id) / f"{image_id}.png"
        return Image.open(path)

    #################################################################
    # Normals: Loading and Storing Utils
    #################################################################

    def sketches_dir_path(self, obj_id: str) -> Path:
        return self.data_dir / "shapes" / obj_id / "sketches"

    def save_sketch(self, normals: np.ndarray, obj_id: str, image_id: str):
        path = self.sketches_dir_path(obj_id) / f"{image_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(normals).save(path)

    def load_sketch(self, obj_id: str, image_id: str) -> Path:
        path = self.sketches_dir_path(obj_id) / f"{image_id}.png"
        return Image.open(path)
