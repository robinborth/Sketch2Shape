import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import pandas as pd
from PIL import Image

from lib.utils.logger import create_logger

logger = create_logger("metainfo")


class MetaInfo:
    def __init__(self, data_dir: str = "data/", split: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.metainfo_path = self.data_dir / "metainfo.csv"
        try:
            metainfo = pd.read_csv(self.metainfo_path)
            if split is not None:
                if split in ["train_latent", "val_latent"]:  # optimize direct latent
                    val_count = (metainfo["split"] == "val").sum()
                    metainfo = metainfo[metainfo["split"] == "train"]
                    split_idx = len(metainfo) - val_count
                    if split == "train_latent":
                        metainfo = metainfo[:split_idx]
                    else:
                        metainfo = metainfo[split_idx:]
                else:
                    metainfo = metainfo[metainfo["split"] == split]
            self._obj_ids = metainfo["obj_id"].to_list()
            self._labels = metainfo["label"].to_list()
            self._obj_id_to_label = {o: l for o, l in zip(self._obj_ids, self._labels)}
            self._label_to_obj_id = {l: o for o, l in zip(self._obj_ids, self._labels)}
        except Exception as e:
            logger.error("Not able to load dataset_splits file.")

        self.image_type_2_type_idx = {
            "sketch": 0,
            "normal": 1,
        }
        self.type_idx_2_image_type = {
            0: "sketch",
            1: "normal",
        }
        self.mode_2_image_type = {
            0: "sketch",
            1: "normal",
            2: "sketch",
            3: "normal",
        }
        self.mode_2_image_dir = {
            0: "sketches",
            1: "normals",
            2: "rendered_sketches",
            3: "rendered_normals",
        }

    #################################################################
    # SNN pairs loader utils
    #################################################################

    def type_idx_2_image_dir(self, type_idx: int, obj_id: str):
        if type_idx == 0:
            return self.data_dir / "shapes" / obj_id / "sketches"
        if type_idx == 1:
            return self.data_dir / "shapes" / obj_id / "normals"
        if type_idx == 2:
            return self.data_dir / "shapes" / obj_id / "rendered_sketches"
        if type_idx == 3:
            return self.data_dir / "shapes" / obj_id / "rendered_normals"
        raise NotImplementedError()

    def iterate_image_data(self, mode: int):
        for obj_id, label in self._obj_id_to_label.items():
            image_dir = self.data_dir / "shapes" / obj_id / self.mode_2_image_dir[mode]
            assert image_dir.exists()
            for file_name in sorted(image_dir.iterdir()):
                yield dict(
                    obj_id=obj_id,
                    image_id=file_name.stem,
                    label=label,
                    image_type=self.mode_2_image_type[mode],
                )

    def load_loss(self, modes: list[int] = [0, 1]):
        data = []
        for mode in modes:
            for image_data in self.iterate_image_data(mode=mode):
                data.append(image_data)
        self._loss_data = pd.DataFrame(data)

    @property
    def loss_count(self):
        return len(self._loss_data)

    @property
    def loss_labels(self):
        return np.array(self._loss_data["label"])

    def get_loss(self, index: int):
        return self._loss_data.iloc[index].to_dict()

    #################################################################
    # Object IDs and Labels
    #################################################################

    @property
    def obj_ids(self):
        return self._obj_ids

    @property
    def obj_id_count(self):
        return len(self.obj_ids)

    def label_to_obj_id(self, label: int) -> str:
        return self._label_to_obj_id.get(label, None)

    def obj_id_to_label(self, obj_id: str) -> int:
        return self._obj_id_to_label.get(obj_id, -1)

    #################################################################
    # Mesh: Loading and Storing Utils
    #################################################################

    def mesh_path(self, obj_id: str) -> Path:
        return self.data_dir / "shapes" / obj_id / "mesh.obj"

    def load_mesh(self, obj_id: str) -> o3d.geometry.TriangleMesh:
        path = self.mesh_path(obj_id)
        return o3d.io.read_triangle_mesh(str(path))

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

    def load_normal(self, obj_id: str, image_id: str):
        path = self.normals_dir_path(obj_id) / f"{image_id}.png"
        return Image.open(path)

    #################################################################
    # Rendered Normals: Loading and Storing Utils
    #################################################################

    def rendered_normals_dir_path(self, obj_id: str) -> Path:
        return self.data_dir / "shapes" / obj_id / "rendered_normals"

    def save_rendered_normal(self, normals: np.ndarray, obj_id: str, image_id: str):
        path = self.rendered_normals_dir_path(obj_id) / f"{image_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(normals).save(path)

    def load_rendered_normal(self, obj_id: str, image_id: str):
        path = self.rendered_normals_dir_path(obj_id) / f"{image_id}.png"
        return Image.open(path)

    #################################################################
    # Sketches: Loading and Storing Utils
    #################################################################

    def sketches_dir_path(self, obj_id: str) -> Path:
        return self.data_dir / "shapes" / obj_id / "sketches"

    def save_sketch(self, normals: np.ndarray, obj_id: str, image_id: str):
        path = self.sketches_dir_path(obj_id) / f"{image_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(normals).save(path)

    def load_sketch(self, obj_id: str, image_id: str):
        path = self.sketches_dir_path(obj_id) / f"{image_id}.png"
        return Image.open(path)

    #################################################################
    # Rendered Sketches: Loading and Storing Utils
    #################################################################

    def rendered_sketches_dir_path(self, obj_id: str) -> Path:
        return self.data_dir / "shapes" / obj_id / "rendered_sketches"

    def save_rendered_sketch(self, normals: np.ndarray, obj_id: str, image_id: str):
        path = self.rendered_sketches_dir_path(obj_id) / f"{image_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(normals).save(path)

    def load_rendered_sketch(self, obj_id: str, image_id: str):
        path = self.rendered_sketches_dir_path(obj_id) / f"{image_id}.png"
        return Image.open(path)

    #################################################################
    # Rendered Config: Loading and Storing Utils
    #################################################################

    def rendered_config_path(self, obj_id: str) -> Path:
        return self.data_dir / "shapes" / obj_id / "rendered_config.csv"

    def save_rendered_config(self, obj_id: str, config: pd.DataFrame):
        path = self.rendered_config_path(obj_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        config.to_csv(path, index=False)

    def load_rendered_config(self, obj_id: str) -> pd.DataFrame:
        path = self.rendered_config_path(obj_id)
        return pd.read_csv(path)

    #################################################################
    # Rendered Latents: Loading and Storing Utils
    #################################################################

    def rendered_latents_path(self, obj_id: str) -> Path:
        return self.data_dir / "shapes" / obj_id / "rendered_latents.npy"

    def save_rendered_latents(self, obj_id: str, latents: np.ndarray):
        path = self.rendered_latents_path(obj_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, latents.astype(np.float32))

    def load_rendered_latents(self, obj_id: str) -> pd.DataFrame:
        path = self.rendered_latents_path(obj_id)
        return np.load(path).astype(np.float32)

    #################################################################
    # General: Loading and Storing Utils
    #################################################################

    def load_image(self, label: int, image_id: int, type_idx: int):
        obj_id = self.label_to_obj_id(label)
        image_type = self.type_idx_2_image_type[type_idx]
        if image_type == "sketch":
            return self.load_sketch(obj_id, f"{image_id:05}")
        if image_type == "normal":
            return self.load_normal(obj_id, f"{image_id:05}")
        if image_type == "rendered_normal":
            return self.load_rendered_normal(obj_id, f"{image_id:05}")
        if image_type == "rendered_sketch":
            return self.load_rendered_sketch(obj_id, f"{image_id:05}")
        raise ValueError(f"Please provide an {image_type=} that is correct!")
