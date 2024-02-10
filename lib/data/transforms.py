import torch
from torchvision.transforms import v2


class SiameseTransform:
    def __init__(
        self,
        normalize: bool = True,
        to_image: bool = True,
        size: int = 256,
        sharpness: float = 1.0,
        mean: float = 0.5,
        std: float = 0.5,
    ):

        transforms = []
        if to_image:
            transforms += v2.ToImage()
        transforms += [
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(size, size), antialias=True),
            v2.RandomAdjustSharpness(sharpness, p=1.0),
        ]
        if normalize:
            transforms += v2.Normalize(mean=[mean], std=[std])

        self.transform = v2.Compose(transforms)

    def __call__(self, image):
        return self.transform(image)


class SketchTransform:
    def __init__(
        self,
        size: list[int] = [64, 128, 256],
        size_weight: list[float] = [0.3, 0.3, 0.4],
        sharpness: list[float] = [0.5, 1, 2, 5],
        sharpness_weight: list[float] = [0.1, 0.3, 0.3, 0.3],
        rotation: int = 5,
        mean: float = 0.5,
        std: float = 0.5,
    ):
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomRotation(degrees=rotation, fill=1.0),
                v2.RandomChoice(
                    [v2.Resize(size=(s, s), antialias=True) for s in size],
                    p=size_weight,
                ),
                v2.RandomChoice(
                    [v2.RandomAdjustSharpness(d, p=1.0) for d in sharpness],
                    p=sharpness_weight,
                ),
                v2.Normalize(mean=[mean], std=[std]),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class NormalTransform:
    def __init__(
        self,
        size: list[int] = [64, 128, 256],
        size_weight: list[float] = [0.3, 0.3, 0.4],
        mean: float = 0.5,
        std: float = 0.5,
    ):
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomChoice(
                    [v2.Resize(size=(s, s), antialias=True) for s in size],
                    p=size_weight,
                ),
                v2.Normalize(mean=[mean], std=[std]),
            ]
        )

    def __call__(self, image):
        return self.transform(image)
