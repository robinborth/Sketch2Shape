import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from lib.data.preprocess import PreprocessMesh, PreprocessSDF, PreprocessSiamese
from lib.utils import create_logger

logger = create_logger("preprocess_data")


@hydra.main(version_base=None, config_path="../conf", config_name="preprocess_data")
def preprocess(cfg: DictConfig) -> None:
    logger.debug("==> initializing mesh ...")
    mesh: PreprocessMesh = hydra.utils.instantiate(cfg.data.preprocess_mesh)
    logger.debug("==> start preprocessing mesh ...")
    for obj_id in tqdm(list(mesh.obj_ids_iter())):
        normalized_mesh = mesh.preprocess(obj_id=obj_id)
        mesh.metainfo.save_normalized_mesh(obj_id=obj_id, mesh=normalized_mesh)

    logger.debug("==> initializing siamese ...")
    siamese: PreprocessSiamese = hydra.utils.instantiate(cfg.data.preprocess_siamese)
    logger.debug("==> start preprocessing siamese ...")
    for obj_id in tqdm(list(siamese.obj_ids_iter())):
        normals, sketches = siamese.preprocess(obj_id=obj_id)
        for idx, (normal, sketch) in enumerate(zip(normals, sketches)):
            siamese.metainfo.save_normal(normal, obj_id=obj_id, image_id=f"{idx:05}")
            siamese.metainfo.save_sketch(sketch, obj_id=obj_id, image_id=f"{idx:05}")

    logger.debug("==> initializing normals everywhere ...")
    normals = hydra.utils.instantiate(cfg.data.preprocess_normals_everywhere)
    logger.debug("==> start preprocessing normals ...")
    for obj_id in tqdm(list(normals.obj_ids_iter())):
        normals = normals.preprocess(obj_id=obj_id)
        for idx, normal in enumerate(normals):
            siamese.metainfo.save_normal_everywhere(normal, obj_id, f"{idx:05}")

    logger.debug("==> initializing sdf ...")
    sdf: PreprocessSDF = hydra.utils.instantiate(cfg.data.preprocess_sdf)
    logger.debug("==> start preprocessing sdf ...")
    for obj_id in tqdm(list(sdf.obj_ids_iter())):
        sdf_samples, surface_samples = sdf.preprocess(obj_id=obj_id)
        sdf.metainfo.save_sdf_samples(obj_id=obj_id, samples=sdf_samples)
        sdf.metainfo.save_surface_samples(obj_id=obj_id, samples=surface_samples)


if __name__ == "__main__":
    preprocess()
