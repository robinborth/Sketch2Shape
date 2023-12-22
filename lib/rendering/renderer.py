from dataclasses import dataclass
from typing import Any, Union

import torch
from torch.nn.functional import normalize

from lib.models.deepsdf import DeepSDF
from lib.rendering.utils import R_azim_elev, dot


@dataclass
class SphereTracer:
    max_steps: int = 50
    warmup_steps: int = 10
    cold_step_scale: float = 0.6
    warm_step_scale: float = 1.0
    surface_threshold: float = 1e-03
    void_threshold: float = 2.0

    def trance(
        self,
        points: torch.Tensor,
        rays: torch.Tensor,
        sdf: Any,
    ):
        points = points.clone()
        total_points = (points.shape[0],)
        device = sdf.device

        depth = torch.zeros(total_points).to(device)
        sd = torch.ones(total_points).to(device)
        mask = torch.full(total_points, True, dtype=torch.bool).to(device)
        surface_mask = torch.full(total_points, False, dtype=torch.bool).to(device)
        void_mask = torch.full(total_points, False, dtype=torch.bool).to(device)

        # sphere tracing
        for step in range(self.max_steps):
            with torch.no_grad():
                sd_out = sdf.predict(points=points, mask=mask)

            step_scale = self.cold_step_scale
            if step > self.warm_step_scale:
                step_scale = self.warm_step_scale

            depth[mask] += sd_out * step_scale
            sd[mask] = sd_out * step_scale

            surface_idx = sd < self.surface_threshold
            mask[surface_idx] = False
            surface_mask[surface_idx] = True

            void_idx = depth > self.void_threshold
            mask[void_idx] = False
            void_mask[void_idx] = True

            points[mask] = points[mask] + sd[mask, None] * rays[mask]

        return points, surface_mask, void_mask


class SignedDistanceFunction:
    def __init__(
        self,
        ckpt_path: str = "/checkpoints/last.ckpt",
        obj_idx: Any = None,
        device: Union[str, None] = None,
    ):
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DeepSDF.load_from_checkpoint(ckpt_path).eval().to(self.device)

        if obj_idx is not None:
            obj_idx = torch.tensor(obj_idx).to(self.device)
            self.lat_vec = self.model.lat_vecs(obj_idx)[None]
        else:
            self.lat_vec = ...

        # self.lat_vec = torch.autograd.Variable(
        #     self.model.lat_vecs(obj_idx)[None], requires_grad=True
        # )

    def predict(self, points, mask=None):
        total_points = points.shape[0]
        lat_vec = self.lat_vec.expand((total_points, -1))

        if mask is not None:
            return self.model.predict((points[mask], lat_vec[mask])).squeeze()
        return self.model.predict((points, lat_vec)).squeeze()


class Camera:
    def __init__(
        self,
        azim: float = 0.0,
        elev: float = 0.0,
        resolution: int = 256,
        dist: float = 1.0,
        focal_length: float = 1.0,
        device: Union[str, None] = None,
    ):
        # rotation matrix
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        R = R_azim_elev(azim=azim, elev=elev)

        # image plane
        xs = torch.linspace(0.5, -0.5, resolution)
        ys = torch.linspace(0.5, -0.5, resolution)
        zs = torch.full((resolution, resolution), -(dist), dtype=torch.float32)
        grid = torch.meshgrid(xs, ys)
        _image_plane = torch.stack([grid[0], grid[1], zs], dim=-1).view(-1, 3)
        self.image_plane = (R @ _image_plane.T).T.to(device)

        # camera point
        _camera = torch.tensor([0, 0, -(dist + focal_length)], dtype=torch.float32)
        self.camera_point = (R @ _camera).to(device)

        # rays
        self.rays = normalize(self.image_plane - self.camera_point)

        self.resolution = resolution


class Light:
    def __init__(
        self,
        position: list[float] = [0, 0, 0],
        normal_step_eps=8e-03,
        ambient: Union[float, list[float]] = 0.5,
        diffuse: Union[float, list[float]] = 0.3,
        specular: Union[float, list[float]] = 0.3,
        shininess: float = 200.0,
        device: Union[str, None] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.position = torch.tensor(position, dtype=torch.float32).to(device)
        self.ambient = torch.tensor(ambient, dtype=torch.float32).to(device)
        self.diffuse = torch.tensor(diffuse, dtype=torch.float32).to(device)
        self.specular = torch.tensor(specular, dtype=torch.float32).to(device)
        self.shininess = shininess
        self.normal_step_eps = normal_step_eps

    def shade(self, points, mask, camera_point, normals):
        N = normals
        L = normalize(self.position - points)
        V = normalize(camera_point - points)

        image = torch.zeros_like(N)
        image += self.ambient
        image += self.diffuse * dot(L, N)
        image += self.specular * torch.pow(dot(N, normalize(L + V)), self.shininess / 4)
        image[~mask] = 1
        return torch.clip(image, 0, 1)


class Scene:
    def __init__(
        self,
        sdf: SignedDistanceFunction,
        camera: Camera,
        light: Light,
        sphere_tracer: SphereTracer = None,
    ):
        self.sdf = sdf
        self.camera = camera
        self.light = light
        self.sphere_tracer = sphere_tracer or SphereTracer()

    def _to_image(self, x, mask, default=0):
        resolution = self.camera.resolution
        x[~mask] = default
        x = x.view(resolution, resolution, -1)
        return x.permute(1, 0, 2)

    def _normals(self, points: torch.Tensor, mask: torch.Tensor):
        points.requires_grad = True
        sd = self.sdf.predict(points, mask)
        loss = sd.sum()
        loss.backward(retain_graph=True)
        return normalize(points.grad)  # type: ignore

    def sphere_tracing(self):
        return self.sphere_tracer.trance(
            points=self.camera.image_plane,
            rays=self.camera.rays,
            sdf=self.sdf,
        )

    def render_depth(self):
        points, surface_mask, _ = self.sphere_tracing()
        depth = torch.linalg.norm((points - self.camera.image_plane), dim=-1)
        return self._to_image(depth, surface_mask)

    def render_normals(self):
        points, surface_mask, _ = self.sphere_tracing()
        normals = self._normals(points, surface_mask)
        return self._to_image(normals, surface_mask), surface_mask

    def render_image(self):
        points, surface_mask, _ = self.sphere_tracing()
        normals = self._normals(points, surface_mask)
        image = self.light.shade(
            points=points,
            mask=surface_mask,
            camera_point=self.camera.camera_point,
            normals=normals,
        )
        return self._to_image(image, surface_mask, default=1)
