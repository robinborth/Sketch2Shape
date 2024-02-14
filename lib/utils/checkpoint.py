from pathlib import Path

from lightning import LightningModule

from lib.models.barlow_twins import BarlowTwins
from lib.models.clip import CLIP
from lib.models.deepsdf import DeepSDF
from lib.models.resnet import ResNet18
from lib.models.siamese import Siamese


def load_model(ckpt_path: str) -> LightningModule:
    path = Path(ckpt_path)

    # first try to load the correct model
    if path.stem == "resnet18":
        return ResNet18()
    if path.stem == "clip":
        return CLIP()
    if path.stem == "siamese" or "train_siamese" in ckpt_path:
        return Siamese.load_from_checkpoint(path)
    if path.stem == "barlow_twins" or "train_barlow_twins" in ckpt_path:
        return BarlowTwins.load_from_checkpoint(path)
    if path.stem == "deepsdf" or "train_deepsdf" in ckpt_path:
        return DeepSDF.load_from_checkpoint(path)

    # try to load the models with some arbirary path
    try:
        return Siamese.load_from_checkpoint(path)
    except Exception:
        pass

    try:
        return BarlowTwins.load_from_checkpoint(path)
    except Exception:
        pass

    try:
        return DeepSDF.load_from_checkpoint(path)
    except Exception:
        pass

    raise FileNotFoundError(f"The provided {ckpt_path=} can not be found!")
