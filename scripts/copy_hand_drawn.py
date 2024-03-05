from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm

from lib.utils.logger import create_logger

logger = create_logger("copy_hand_drawn")


# python scripts/copy_hand_drawn.py +source=data/shapenet_chair_hand_drawn
@hydra.main(version_base=None, config_path="../conf", config_name="preprocess_data")
def main(cfg: DictConfig) -> None:
    logger.debug("==> loading config ...")
    L.seed_everything(cfg.seed)

    logger.debug("==> copy and transform hand drawn dataset ...")
    padding = 0.05
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    final_transform = v2.Compose([v2.Resize((256, 256)), v2.ToPILImage()])
    shapes_path = Path(cfg.data.data_dir) / "shapes"
    total = len(list(Path(cfg.source).iterdir()))
    for path in tqdm(Path(cfg.source).iterdir(), total=total):
        obj_id = path.stem
        image = Image.open(path)

        # extract the sketch
        sketch = transform(image)
        mask = sketch.sum(0) == 0.0
        idx = np.where(mask)
        try:
            bbox = sketch[
                :, np.min(idx[0]) : np.max(idx[0]), np.min(idx[1]) : np.max(idx[1])
            ]
        except Exception:
            logger.error(f"Something is wrong with {obj_id} ...")
            bbox = sketch

        # add padding
        max_size = max(bbox.shape[1], bbox.shape[2])
        pad_2 = (max_size - bbox.shape[2]) // 2
        pad_1 = (max_size - bbox.shape[1]) // 2
        bbox = torch.nn.functional.pad(bbox, (pad_2, pad_2, pad_1, pad_1), value=1.0)
        margin = int(max_size * padding)
        bbox = torch.nn.functional.pad(
            bbox, (margin, margin, margin, margin), value=1.0
        )

        hand_drawn = final_transform(bbox)
        out_path = shapes_path / obj_id / "eval_hand_drawn" / "00000.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        hand_drawn.save(out_path)


if __name__ == "__main__":
    main()
