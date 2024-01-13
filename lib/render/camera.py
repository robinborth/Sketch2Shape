import torch

from lib.render.utils import get_rotation_x, get_rotation_y, get_translation


class Camera:
    def __init__(
        self,
        azim: float = 0.0,
        elev: float = 0.0,
        dist: float = 4.0,
        width: int = 256,
        height: int = 256,
        focal: int = 512,
        sphere_eps: float = 3e-2,
    ):
        self.azim = azim
        self.elev = elev
        self.dist = dist
        self.width = width
        self.height = height
        self.focal = focal
        self.sphere_eps = sphere_eps

    def get_world_to_camera(self):
        mat = get_translation(self.dist)
        mat = mat @ get_rotation_x(self.elev)
        mat = mat @ get_rotation_y(self.azim)
        return mat

    def get_camera_to_world(self):
        return torch.inverse(self.get_world_to_camera())

    def camera_position(self):
        return self.get_camera_to_world()[:3, -1]

    def rays(self):
        P = self.get_camera_to_world()

        # Screen coordinates
        pixel_xs, pixel_ys = torch.meshgrid(
            torch.arange(self.width), torch.arange(self.height)
        )
        xs = (pixel_xs - self.width * 0.5) / self.focal
        ys = (pixel_ys - self.height * 0.5) / self.focal

        # homogeneous coordinates
        coords = [-xs, -ys, torch.ones_like(pixel_xs), torch.ones_like(pixel_xs)]
        image_plane_camera = torch.stack(coords, axis=-1)

        image_plane_world = P @ image_plane_camera.view(-1, 4).T
        image_plane_world = image_plane_world.T.view(self.width, self.height, 4)
        image_plane_world_coord = image_plane_world[:, :, :3]

        points = P[:3, -1].expand(image_plane_world_coord.shape)
        rays = torch.nn.functional.normalize(image_plane_world_coord - points, dim=-1)

        # HACK fix the width, height indexing
        points = points.permute(1, 0, 2)
        rays = rays.permute(1, 0, 2)

        return points, rays

    def unit_sphere_intersection_rays(self):
        points, rays = self.rays()
        points = points.reshape(-1, 3)
        rays = rays.reshape(-1, 3)
        radius = self.sphere_eps + 1.0

        # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection.html
        L = -points
        t_ca = (L * rays).sum(dim=-1)
        d = torch.sqrt((L * L).sum(dim=-1) - t_ca**2)
        t_hc = torch.sqrt(radius**2 - d**2)
        t_hc = torch.nan_to_num(t_hc, -1)
        mask = (t_ca >= 0) & (t_hc >= 0)

        depth_0 = t_ca - t_hc
        depth_0[~mask] = 0

        depth_1 = t_ca + t_hc
        depth_1[~mask] = 0

        points = points + rays * depth_0[..., None]
        return points, rays, mask
