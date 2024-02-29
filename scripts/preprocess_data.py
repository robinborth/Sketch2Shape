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

    logger.debug("==> initializing synthethic sketches ...")
    synthetic = hydra.utils.instantiate(cfg.data.preprocess_synthetic)()
    metainfo = synthetic.metainfo
    logger.debug("==> start preprocessing synthethic sketches ...")
    for obj_id in tqdm(list(synthetic.obj_ids_iter())):
        norms, sketchs, grays = synthetic.preprocess(obj_id)
        for idx, (normal, sketch, grayscale) in enumerate(zip(norms, sketchs, grays)):
            metainfo.save_normal(normal, obj_id, f"{idx:05}")
            metainfo.save_sketch(sketch, obj_id, f"{idx:05}")
            metainfo.save_synthetic_grayscale(grayscale, obj_id, f"{idx:05}")

    if cfg.get("deepsdf_ckpt_path"):
        for split in ["train_latent", "val_latent"]:
            logger.debug(f"==> initializing renderings for {split} ...")
            cfg.data.preprocess_renderings.split = split
            cfg.data.preprocess_renderings.traversal = False
            renderings = hydra.utils.instantiate(cfg.data.preprocess_renderings)()
            metainfo = renderings.metainfo
            logger.debug(f"==> start preprocessing renderings for {split} ...")
            for obj_id in tqdm(list(renderings.obj_ids_iter())):
                norms, sketchs, grays, latents, config = renderings.preprocess(obj_id)
                for idx, (norm, sketch, gray) in enumerate(zip(norms, sketchs, grays)):
                    metainfo.save_rendered_normal(norm, obj_id, f"{idx:05}")
                    metainfo.save_rendered_sketch(sketch, obj_id, f"{idx:05}")
                    metainfo.save_rendered_grayscale(gray, obj_id, f"{idx:05}")
                metainfo.save_rendered_config(obj_id, config)
                metainfo.save_rendered_latents(obj_id, latents)

    if cfg.get("deepsdf_ckpt_path"):
        for split in ["train_latent", "val_latent"]:
            logger.debug(f"==> initializing traversal for {split} ...")
            cfg.data.preprocess_renderings.split = split
            cfg.data.preprocess_renderings.traversal = True
            cfg.data.preprocess_renderings.n_renderings = 20
            traversals = hydra.utils.instantiate(cfg.data.preprocess_renderings)()
            metainfo = traversals.metainfo
            logger.debug(f"==> start preprocessing traversals for {split} ...")
            for obj_id in tqdm(list(traversals.obj_ids_iter())):
                norms, sketchs, grays, latents, config = traversals.preprocess(obj_id)
                for idx, (norm, sketch, gray) in enumerate(zip(norms, sketchs, grays)):
                    metainfo.save_traversed_normal(norm, obj_id, f"{idx:05}")
                    metainfo.save_traversed_sketch(sketch, obj_id, f"{idx:05}")
                    metainfo.save_traversed_grayscale(gray, obj_id, f"{idx:05}")
                metainfo.save_traversed_config(obj_id, config)
                metainfo.save_traversed_latents(obj_id, latents)


if __name__ == "__main__":
    preprocess()
