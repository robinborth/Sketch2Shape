from pathlib import Path

import hydra
import lightning as L
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from lib.data.metainfo import MetaInfo
from lib.utils import create_logger

logger = create_logger("copy_shapenet")


# python scripts/copy_shapenet.py +source=/shared/data/ShapeNetCore/03001627
@hydra.main(version_base=None, config_path="../conf", config_name="preprocess_data")
def main(cfg: DictConfig) -> None:
    logger.debug("==> loading config ...")
    L.seed_everything(cfg.seed)

    metainfo = MetaInfo(cfg.data.data_dir)

    assert cfg.source
    obj_ids = [obj_file.stem for obj_file in Path(cfg.source).iterdir()]

    num_samples = cfg.data.num_obj_train + cfg.data.num_obj_val + cfg.data.num_obj_test
    obj_ids = obj_ids[:num_samples]
    assert num_samples == len(obj_ids)

    logger.debug(f"==> start copy {num_samples} sketches ... ")
    for obj_id in tqdm(obj_ids):
        source_path = Path(cfg.source) / obj_id / "models/model_normalized.obj"
        metainfo.save_mesh(source_path=source_path, obj_id=obj_id)

    data = []
    splits = (
        ["train"] * cfg.data.num_obj_train
        + ["val"] * cfg.data.num_obj_val
        + ["test"] * cfg.data.num_obj_test
    )
    for label, obj_id in enumerate(obj_ids):
        data.append({"obj_id": obj_id, "label": label, "split": splits[label]})
    df = pd.DataFrame(data)
    df.to_csv(Path(cfg.data.data_dir) / "metainfo.csv", index=False)


if __name__ == "__main__":
    main()
