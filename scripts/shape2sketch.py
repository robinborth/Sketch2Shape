import os
from pathlib import Path

import cv2
import hydra
import lightning as L
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from lib.data.metainfo import MetaInfo
from lib.data.preprocess_sketch import (
    cartesian_elev_azim,
    image_path,
    image_to_sketch,
    images_folder,
    obj_path,
    render_shapenet,
    sketch_path,
    sketches_folder,
)
from lib.utils import create_logger

logger = create_logger("shape2sketch")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.debug("==> loading config ...")
    L.seed_everything(cfg.seed)

    metainfo = MetaInfo(cfg=cfg)
    logger.debug(f"==> start extracting sketches for {metainfo.obj_id_count} shapes...")
    for obj_id in tqdm(metainfo.obj_ids, total=metainfo.obj_id_count):
        shape_path = obj_path(obj_id, config=cfg)
        elevs, azims = cartesian_elev_azim(elev=cfg.elev, azim=cfg.azim)
        images = render_shapenet(
            shape_path,
            dist=cfg.dist,
            color=cfg.color,
            elev=elevs,
            azim=azims,
            image_size=cfg.image_size,
            device=cfg.device,
        )
        sketches = image_to_sketch(
            images,
            t_lower=cfg.t_lower,
            t_upper=cfg.t_upper,
            aperture_size=cfg.aperture_size,
            L2gradient=cfg.L2gradient,
        )
        for index, (image, sketch) in enumerate(zip(images, sketches)):
            if not os.path.exists(images_folder(obj_id, config=cfg)):
                os.mkdir(images_folder(obj_id, config=cfg))
            _image_path = image_path(obj_id, index=index, config=cfg)
            cv2.imwrite(_image_path, image)

            if not os.path.exists(sketches_folder(obj_id, config=cfg)):
                os.mkdir(sketches_folder(obj_id, config=cfg))
            _sketch_path = sketch_path(obj_id, index=index, config=cfg)
            cv2.imwrite(_sketch_path, sketch)

    logger.debug(f"==> creating metainfo for {metainfo.obj_id_count} shapes...")
    data = []
    label = 0
    for obj_id, split in tqdm(metainfo.obj_ids_splits, total=metainfo.obj_id_count):
        sketch_paths = Path(cfg.dataset_path, obj_id, "sketches").glob("*.jpg")
        for sketch_id in sorted(list(path.stem for path in sketch_paths)):
            image_paths = Path(cfg.dataset_path, obj_id, "images").glob("*.jpg")
            for image_id in sorted(list(path.stem for path in image_paths)):
                data.append(
                    {
                        "obj_id": str(obj_id),
                        "sketch_id": str(sketch_id),
                        "image_id": str(image_id),
                        "label": label,
                        "split": str(split),
                    }
                )
        label += 1
    df = pd.DataFrame(data).sample(frac=1.0)
    df.to_csv(cfg.sketch_image_pairs_path, index=None)


if __name__ == "__main__":
    main()
