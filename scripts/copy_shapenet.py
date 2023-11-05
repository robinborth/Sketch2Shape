import os
import random
import shutil
from pathlib import Path

import cv2
import hydra
import lightning as L
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from lib.utils import create_logger

logger = create_logger("copy_shapenet")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.debug("==> loading config ...")
    L.seed_everything(cfg.seed)

    # get the paths to the shapes
    model_paths = list(Path(cfg.shapenet_folder).glob("*/models/model_normalized.obj"))
    total_num = cfg.num_obj_train + cfg.num_obj_val + cfg.num_obj_test
    paths = random.choices(model_paths, k=total_num)

    logger.debug(f"==> start copy {total_num} sketches ... ")
    obj_ids = []
    for source_path in tqdm(paths):
        try:
            # track the object_ids
            obj_id = source_path.parent.parent.stem
            obj_ids.append(obj_id)

            # create the folder
            destination_directory = Path(cfg.dataset_path, obj_id)
            os.makedirs(destination_directory, exist_ok=True)

            # copy the obj file to the folder
            destination_path = Path(destination_directory, source_path.name)
            shutil.copy2(source_path, destination_path)
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")

    logger.debug("==> save the dataset splits ... ")
    splits = (
        ["train"] * cfg.num_obj_train
        + ["val"] * cfg.num_obj_val
        + ["test"] * cfg.num_obj_test
    )
    assert len(splits) == len(paths)
    df = pd.DataFrame({"obj_id": obj_ids, "split": splits})
    df.to_csv(cfg.dataset_splits_path, index=None)


if __name__ == "__main__":
    main()
