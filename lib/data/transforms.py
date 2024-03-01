import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform


class BaseTransform:
    def __init__(
        self,
        to_image: bool = True,
        to_dtype: bool = True,
        normalize: bool = True,
        mean: float = 0.5,
        std: float = 0.5,
        transforms: list = [],
    ):
        transform = []
        if to_image:
            transform.append(v2.ToImage())
        if to_dtype:
            transform.append(v2.ToDtype(torch.float32, scale=True))

        # add custom transformations
        transform.extend(transforms)

        if normalize:
            transform.append(v2.Normalize(mean=[mean], std=[std]))

        self.transform = v2.Compose(transform)

    def __call__(self, image):
        return self.transform(image)


class SketchTransform(BaseTransform):
    def __init__(self, normalize: bool = True):
        transforms = [v2.Resize((256, 256)), ToSketch(), DilateSketch(kernel_size=5)]
        # transforms = [v2.Resize((256, 256))]
        super().__init__(normalize=normalize, transforms=transforms)


############################################################
# Custom Transforms Layers
############################################################


class ToSketch(Transform):
    """Convert the image to an sketch.

    The input of the sketch needs to be of dim 3xHxW and the output
    """

    def __init__(
        self,
        t_lower: int = 100,
        t_upper: int = 150,
        aperture_size: int = 3,  # 3, 5, 7
        l2_gradient: bool = True,
    ):
        super().__init__()
        self.t_lower = t_lower
        self.t_upper = t_upper
        self.aperture_size = aperture_size
        self.l2_gradient = l2_gradient

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


class DilateSketch(Transform):
    def __init__(self, kernel_size: int = 1):
        super().__init__()
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
        with torch.no_grad():
            img = self.conv(img)
        img = 1.0 - torch.min(img, torch.tensor(1.0))
        img = v2.functional.resize(img, (H, W), antialias=True)
        return torch.clip(img, 0.0, 1.0)


class ToSilhouette(Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, image):
        surface_maks = image.sum(0) < 2.95
        image[:, surface_maks] = 0.0
        return image


############################################################
# Deprecated Transforms
############################################################


class ToGrayScale(Transform):
    pass
