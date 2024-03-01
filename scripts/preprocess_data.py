import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from lib.utils.logger import create_logger

logger = create_logger("preprocess_data")


@hydra.main(version_base=None, config_path="../conf", config_name="preprocess_data")
def preprocess(cfg: DictConfig) -> None:
    logger.debug("==> initializing mesh ...")
    mesh = hydra.utils.instantiate(cfg.data.preprocess_mesh)()
    logger.debug("==> start preprocessing mesh ...")
    for obj_id in tqdm(list(mesh.obj_ids_iter())):
        normalized_mesh = mesh.preprocess(obj_id)
        mesh.metainfo.save_normalized_mesh(obj_id, normalized_mesh)

    logger.debug("==> initializing sdf ...")
    sdf = hydra.utils.instantiate(cfg.data.preprocess_sdf)()
    logger.debug("==> start preprocessing sdf ...")
    for obj_id in tqdm(list(sdf.obj_ids_iter())):
        sdf_samples, surface_samples = sdf.preprocess(obj_id)
        sdf.metainfo.save_sdf_samples(obj_id, sdf_samples)
        sdf.metainfo.save_surface_samples(obj_id, surface_samples)

    # logger.debug("==> initializing synthethic images ...")
    # synthetic = hydra.utils.instantiate(cfg.data.preprocess_synthetic)()
    # logger.debug("==> start preprocessing synthethic images ...")
    # for obj_id in tqdm(list(synthetic.obj_ids_iter())):
    #     norms, sketchs, grays = synthetic.preprocess(obj_id)
    #     for image_id, (norm, sketch, gray) in enumerate(zip(norms, sketchs, grays)):
    #         synthetic.metainfo.save_image(obj_id, sketch, image_id, 0)
    #         synthetic.metainfo.save_image(obj_id, norm, image_id, 1)
    #         synthetic.metainfo.save_image(obj_id, gray, image_id, 2)

    if cfg.get("deepsdf_ckpt_path") is None:
        return

    for split in ["train_latent", "val_latent"]:
        logger.debug(f"==> initializing renderings for {split} ...")
        cfg.data.preprocess_renderings.split = split
        render = hydra.utils.instantiate(cfg.data.preprocess_renderings)()
        logger.debug(f"==> start preprocessing renderings for {split} ...")
        for obj_id in tqdm(list(render.obj_ids_iter())):
            norms, sketchs, grays, latents, config = render.preprocess(obj_id)
            for image_id, (norm, sketch, gray) in enumerate(zip(norms, sketchs, grays)):
                render.metainfo.save_image(obj_id, sketch, image_id, 3)
                render.metainfo.save_image(obj_id, norm, image_id, 4)
                render.metainfo.save_image(obj_id, gray, image_id, 5)
            render.metainfo.save_config(obj_id, config, 3)
            render.metainfo.save_latents(obj_id, latents, 3)

    for split in ["train_latent", "val_latent"]:
        logger.debug(f"==> initializing traversal for {split} ...")
        cfg.data.preprocess_traversal.split = split
        traverse = hydra.utils.instantiate(cfg.data.preprocess_traversal)()
        logger.debug(f"==> start preprocessing traversals for {split} ...")
        for obj_id in tqdm(list(traverse.obj_ids_iter())):
            norms, sketchs, grays, latents, config = traverse.preprocess(obj_id)
            for image_id, (norm, sketch, gray) in enumerate(zip(norms, sketchs, grays)):
                traverse.metainfo.save_image(obj_id, sketch, image_id, 6)
                traverse.metainfo.save_image(obj_id, norm, image_id, 7)
                traverse.metainfo.save_image(obj_id, gray, image_id, 8)
            traverse.metainfo.save_config(obj_id, config, 6)
            traverse.metainfo.save_latents(obj_id, latents, 6)


if __name__ == "__main__":
    preprocess()
