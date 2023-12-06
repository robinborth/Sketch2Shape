import os
from pathlib import Path

import cv2
import h5py
import hydra
import lightning as L
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from lib.data.metainfo import MetaInfo
from lib.utils import create_logger

logger = create_logger("create_h5py")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.debug("==> loading config ...")
    L.seed_everything(cfg.seed)

    metainfo = MetaInfo(data_dir=cfg.data.data_dir)
    logger.debug(f"==> start creating h5py for {metainfo.obj_id_count} shapes...")
    for obj_id in tqdm(metainfo.obj_ids, total=metainfo.obj_id_count):
        paths = Path(cfg.data.data_dir, obj_id, "images").glob("*.jpg")
        images = np.stack([cv2.imread(path.as_posix()) for path in paths])
        paths = Path(cfg.data.data_dir, obj_id, "sketches").glob("*.jpg")
        sketches = np.stack([cv2.imread(path.as_posix()) for path in paths])
        h5_file = h5py.File(
            Path(cfg.data.data_dir, obj_id, cfg.data.hd5py_file_name), "w"
        )
        h5_file.create_dataset("sketches", data=sketches, compression="gzip")
        h5_file.create_dataset("images", data=images, compression="gzip")
        # h5_file.create_dataset(
        #     "sketches", np.shape(sketches), h5py.h5t.STD_U8BE, data=sketches
        # )
        # h5_file.create_dataset(
        #     "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
        # )
        h5_file.close()


if __name__ == "__main__":
    main()
