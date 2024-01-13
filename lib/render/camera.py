import numpy as np

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
        sphere_eps: float = 1e-1,
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
        return np.linalg.inv(self.get_world_to_camera())

    def camera_position(self):
        return self.get_camera_to_world()[:3, -1]

    def rays(self):
        P = self.get_camera_to_world()

        # Screen coordinates
        pixel_xs, pixel_ys = np.meshgrid(np.arange(self.width), np.arange(self.height))
        xs = (pixel_xs - self.width * 0.5) / self.focal
        ys = (pixel_ys - self.height * 0.5) / self.focal

        # homogeneous coordinates
        coords = [-xs, -ys, np.ones_like(pixel_xs), np.ones_like(pixel_xs)]
        image_plane_camera = np.stack(coords, axis=-1)

        image_plane_world = P @ image_plane_camera.reshape(-1, 4).T
        image_plane_world = image_plane_world.T.reshape(self.width, self.height, 4)
        image_plane_world_coord = image_plane_world[:, :, :3]

        points = [P[:3, -1]] * (self.width * self.height)
        points = np.stack(points).reshape(self.width, self.height, 3)
        rays = image_plane_world_coord - points
        rays = rays / np.linalg.norm(rays, axis=-1)[..., None]

        # HACK fix the width, height indexing
        points = points.astype(np.float32)
        rays = rays.astype(np.float32)

        return points, rays

    def unit_sphere_intersection_rays(self):
        points, rays = self.rays()
        points = points.reshape(-1, 3)
        rays = rays.reshape(-1, 3)
        radius = self.sphere_eps + 1.0

        # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection.html
        L = -points
        t_ca = (L * rays).sum(axis=-1)
        _d = (L * L).sum(axis=-1) - t_ca**2
        d = np.sqrt(np.abs(_d))
        d[_d < 0] = 0

        _t_hc = radius**2 - d**2
        t_hc = np.sqrt(np.abs(_t_hc))
        t_hc[_t_hc < 0] = -1
        mask = (t_ca >= 0) & (t_hc >= 0)

        depth_0 = t_ca - t_hc
        depth_0[~mask] = 0

        depth_1 = t_ca + t_hc
        depth_1[~mask] = 0

        points = points + rays * depth_0[..., None]
        return points, rays, mask
