import os

import cv2
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from lib.data.sketch import (
    cartesian_elev_azim,
    default_elev_azim,
    image_grid,
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
    logger.debug(f"==> start extracting sketches for {len(cfg.obj_ids)} shapes...")
    for obj_id in tqdm(cfg.obj_ids, total=len(cfg.obj_ids)):
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
            cv2.imwrite(_image_path, sketch)

            if not os.path.exists(sketches_folder(obj_id, config=cfg)):
                os.mkdir(sketches_folder(obj_id, config=cfg))
            _sketch_path = sketch_path(obj_id, index=index, config=cfg)
            cv2.imwrite(_sketch_path, sketch)


if __name__ == "__main__":
    main()
