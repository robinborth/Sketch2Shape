from dataclasses import dataclass

import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import v2


class BaseTransform:
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
            transforms.append(v2.ToImage())
        transforms += [
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(size, size), antialias=True),
            v2.RandomAdjustSharpness(sharpness, p=1.0),
        ]
        if normalize:
            transforms.append(v2.Normalize(mean=[mean], std=[std]))

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


@dataclass
class ToSketch(object):
    """Convert the image to an sketch.

    The input of the sketch needs to be of dim 3xHxW and the output
    """

    t_lower: int = 100
    t_upper: int = 150
    aperture_size: int = 3  # 3, 5, 7
    l2_gradient: bool = True

    def __call__(self, image: torch.Tensor):
        """Transforms an image into an sketch.

        Args:
            image (torch.Tensor): An image of dimension 3xHxW with values between (0,1)

        Returns:
            torch.Tensor: The image as an sketch.
        """

        img = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        sketch = cv.Canny(
            img,
            threshold1=self.t_lower,
            threshold2=self.t_upper,
            apertureSize=self.aperture_size,
            L2gradient=self.l2_gradient,
        )
        sketch = cv.bitwise_not(sketch)
        sketch = np.stack((np.stack(sketch),) * 3, axis=-1) / 255
        return torch.tensor(sketch).to(image).permute(2, 0, 1)


@dataclass
class DilateSketch(object):
    def __init__(self, kernel_size: int = 1):
        self.conv = torch.nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=kernel_size,
            padding="same",
            stride=1,
            bias=False,
        )
        self.conv.weight = torch.nn.Parameter(torch.ones_like(self.conv.weight))
        self.padding = (kernel_size - 1) * 2

    def __call__(self, image):
        _, H, W = image.shape
        img = 1.0 - image
        img = v2.functional.pad(img, padding=self.padding)  # 3xH+PxW+P
        img = self.conv(img)
        img = 1.0 - torch.min(img, torch.tensor(1.0))
        img = v2.functional.resize(img, (H, W), antialias=True)
        return torch.clip(img, 0, 1)


@dataclass
class ToSilhouette(object):
    def __call__(self, image):
        surface_maks = image.sum(0) < 2.95
        image[:, surface_maks] = 0.0
        return image


@dataclass
class ToGrayScale(object):
    def __call__(self, image):
        mean = image.mean(0)
        return torch.stack([mean, mean, mean], dim=0)
