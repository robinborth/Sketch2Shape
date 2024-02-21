from typing import Callable, Optional

from torch.utils.data import Dataset

from lib.data.metainfo import MetaInfo
from lib.models.deepsdf import DeepSDF

############################################################
# Base Loss Datasets
############################################################


class LossDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/",
        split: Optional[str] = None,
        modes: list[int] = [0, 1],
        sketch_transform: Optional[Callable] = None,
        normal_transform: Optional[Callable] = None,
    ):
        self.sketch_transform = sketch_transform
        self.normal_transform = normal_transform
        self.metainfo = MetaInfo(data_dir=data_dir, split=split)
        self.metainfo.load_loss(modes=modes)

    def __len__(self):
        return self.metainfo.loss_count

    def fetch(self, index):
        info = self.metainfo.get_loss(index)
        obj_id = info["obj_id"]
        image_id = info["image_id"]
        label = info["label"]
        image_type = self.metainfo.mode_2_image_type[info["mode"]]
        type_idx = self.metainfo.mode_2_type_idx[info["mode"]]

        if image_type == "sketch":
            image = self.metainfo.load_sketch(obj_id, image_id)
            if self.sketch_transform is not None:
                image = self.sketch_transform(image)

        if image_type == "rendered_sketch":
            image = self.metainfo.load_rendered_sketch(obj_id, image_id)
            if self.sketch_transform is not None:
                image = self.sketch_transform(image)

        if image_type == "normal":
            image = self.metainfo.load_normal(obj_id, image_id)
            if self.normal_transform is not None:
                image = self.normal_transform(image)

        if image_type == "rendered_normal":
            image = self.metainfo.load_rendered_normal(obj_id, image_id)
            if self.normal_transform is not None:
                image = self.normal_transform(image)

        return {
            "image": image,
            "image_id": int(image_id),
            "type_idx": type_idx,
            "label": label,
        }

    def __getitem__(self, index: int):
        return self.fetch(index)


############################################################
# Latent Loss Dataset
############################################################


class LatentLossDataset(LossDataset):
    def __init__(self, deepsdf_ckpt_path: str = "deepsdf.ckpt", **kwargs):
        super().__init__(**kwargs)
        self.deepsdf = DeepSDF.load_from_checkpoint(
            checkpoint_path=deepsdf_ckpt_path,
            map_location="cpu",
        )

    def __getitem__(self, index):
        item = self.fetch(index)
        label = item["label"]
        latent = self.deepsdf.lat_vecs.weight[label].detach().cpu().numpy().flatten()
        item["latent"] = latent
        return item
