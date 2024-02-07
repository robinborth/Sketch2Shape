import torch
from torchvision.transforms import v2


class SketchTransform:
    def __init__(
        self,
        mean: float = 0.5,
        std: float = 0.5,
    ):
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[mean], std=[std]),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class NormalTransform:
    def __init__(
        self,
        mean: float = 0.5,
        std: float = 0.5,
    ):
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[mean], std=[std]),
            ]
        )

    def __call__(self, image):
        return self.transform(image)
