from typing import Callable, Optional

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

    def __len__(self):
        return self.metainfo.pair_count

    def __getitem__(self, index):
        info = self.metainfo.get_pair(index)
        obj_id = info["obj_id"]
        image_id = info["image_id"]
        label = info["label"]

        sketch = self.metainfo.load_sketch(obj_id, image_id)
        image = self.metainfo.load_normal(obj_id, image_id)
        if self.transforms is not None:
            sketch = self.transforms(sketch)
            image = self.transforms(image)

        return {
            "sketch": sketch,
            "image": image,
            "label": label,
        }
