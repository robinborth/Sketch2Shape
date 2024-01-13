from typing import Callable, Optional

import cv2
from torch.utils.data import Dataset

from lib.data.metainfo import MetaInfo


class SiameseDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/",
        split: Optional[str] = None,
        transforms: Optional[Callable] = None,
    ):
        self.transforms = transforms
        self.metainfo = MetaInfo(data_dir=data_dir, split=split)
        self.metainfo.load_sketch_image_pairs()

    def _load(self, obj_id: str, render_type: str, image_id: str):
        path = self.metainfo.render_path(
            obj_id=obj_id,
            render_type=render_type,
            image_id=image_id,
        )
        image = cv2.imread(path.as_posix())  # TODO check if this is correct
        if self.transforms is not None:
            return self.transforms(image)
        return image

    def __len__(self):
        return self.metainfo.pair_count

    def __getitem__(self, index):
        info = self.metainfo.get_pair(index)
        obj_id = info["obj_id"]
        image_id = info["image_id"]
        label = info["label"]

        sketch = self._load(obj_id, "sketches", image_id)
        image = self._load(obj_id, "images", image_id)

        return {
            "sketch": sketch,
            "image": image,
            "label": label,
        }
