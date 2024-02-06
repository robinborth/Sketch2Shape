import torch
from torchvision.transforms import v2


class SketchTransform:
    def __init__(
        self,
        image_size: int,
        rotation_degree: int = 15,
        p_zoom_out: float = 0.5,
        p_erasing: float = 0.5,
        p_adjust_sharpness: float = 0.5,
        p_perspective: float = 0.5,
        mean: float = 0.5,
        std: float = 0.5,
    ):
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                # v2.ToDtype(torch.uint8, scale=True),
                # v2.RandomErasing(p=p_erasing, value=255),
                # v2.Compose(
                #     [
                #         v2.RandomZoomOut(p=p_zoom_out, fill=255),
                #         v2.Resize((image_size, image_size), antialias=True),
                #     ]
                # ),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomRotation(degrees=rotation_degree, fill=1.0),
                v2.RandomAdjustSharpness(10, p=p_adjust_sharpness),
                # v2.RandomPerspective(
                #     p=p_perspective,
                #     fill=1.0,
                #     interpolation=v2.InterpolationMode.NEAREST,
                # ),
                v2.Normalize(mean=[mean], std=[std]),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class NormalTransform:
    def __init__(
        self,
        image_size: int,
        p_zoom_out: float = 0.5,
        p_erasing: float = 0.5,
        mean: float = 0.5,
        std: float = 0.5,
    ):
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                # v2.RandomErasing(p=p_erasing, value=255),
                # v2.Compose(
                #     [
                #         v2.RandomZoomOut(p=p_zoom_out, fill=255),
                #         v2.Resize((image_size, image_size), antialias=True),
                #     ]
                # ),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[mean], std=[std]),
            ]
        )

    def __call__(self, image):
        return self.transform(image)
