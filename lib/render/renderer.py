# some parts inspired by: https://omaraflak.medium.com/ray-tracing-from-scratch-in-python-41670e6a96f9
from typing import Optional

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch


class LightSource:
    def __init__(self, position, reflection):
        self.position = position
        self.reflection = reflection
        self.ambient = reflection.ambient
        self.diffuse = reflection.diffuse
        self.specular = reflection.specular


# for Blinn-Phong model
# this is an object without any fucntion just holding variables - how is that done is correct software engineering desing pattern?
class ReflectionProperty:
    def __init__(
        self,
        ambient: torch.Tensor = torch.tensor([0.1, 0, 0]),
        diffuse: torch.Tensor = torch.tensor([0.7, 0, 0]),
        specular: torch.Tensor = torch.tensor([1, 1, 1]),
        shininess: Optional[int] = 100,
    ):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess


def normalize(vector):
    return vector / vector.norm(dim=-1, keepdim=True)


def sphere_intersection(ray_origin, ray_direction, eps=0.03):
    # unit sphere
    center = torch.tensor([0, 0, 0])
    radius = torch.tensor(1 + eps)

    ray_origin = ray_origin.reshape(-1, 3)
    ray_direction = ray_direction.reshape(-1, 3)

    b = (ray_direction * (ray_origin - center)) * 2
    b = b.sum(1)

    c = torch.norm(ray_origin - center, dim=1) ** 2 - radius**2

    delta = b**2 - 4 * c

    t1 = (-b + delta.sqrt()) / 2
    t2 = (-b - delta.sqrt()) / 2
    return torch.nan_to_num(torch.min(t1, t2))


# following 4 functions inspired by: https://keras.io/examples/vision/nerf/
def get_translation_t(t):
    mat = torch.eye(4)
    mat[2][3] += t
    return mat


# TODO understand rotation matrix
# [ ] play around and see what rotation changes the scene in what way
# [x] check why we need to transpose our image (self.image = depth.view(self.height, self.width).T)
# [ ] understand where the image plane lies
# rotatation matrix arround axis
# first rotate around x axis, then z axis
def get_rotation_x(deg):
    deg = torch.deg2rad(torch.tensor(deg))
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, torch.cos(deg), -torch.sin(deg), 0],
            [0, torch.sin(deg), torch.cos(deg), 0],
            [0, 0, 0, 1],
        ]
    )


def get_rotation_y(deg):
    deg = torch.deg2rad(torch.tensor(deg))
    return torch.tensor(
        [
            [torch.cos(deg), 0, torch.sin(deg), 0],
            [0, 1, 0, 0],
            [-torch.sin(deg), 0, torch.cos(deg), 0],
            [0, 0, 0, 1],
        ]
    )


def get_rotation_z(deg):
    deg = torch.deg2rad(torch.tensor(deg))
    return torch.tensor(
        [
            [torch.cos(deg), -torch.sin(deg), 0, 0],
            [torch.sin(deg), torch.cos(deg), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def get_world_to_camera(azim, elev, dist):
    mat = get_translation_t(dist)
    mat = mat @ get_rotation_x(elev)
    mat = mat @ get_rotation_y(azim)
    return mat


def get_camera_to_world(azim, elev, dist):
    return torch.inverse(get_world_to_camera(azim, elev, dist))


def get_camera_loc(R, t):
    return -R.T @ t


def get_camera_rays(R):
    return R.T @ torch.tensor([0, 0, 1.0])


# TODO
# [ ] change all the [:, mask.squeeze()] to just [mask], it should work out


# inspired by https://omaraflak.medium.com/ray-tracing-from-scratch-in-python-41670e6a96f9
class Renderer:
    # TODO
    # [ ] make most of the used variables class/object variables instead of function passes
    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        focal: int = 512,
        surface_eps: float = 1e-5,
        sphere_eps: float = 3e-2,
        lightsource: Optional[LightSource] = None,
    ):
        self.width = width
        self.height = height
        self.focal = focal
        self.surface_eps = surface_eps
        self.sphere_eps = sphere_eps
        self.lightsource = lightsource

        # TODO define function get_img_to_camera()
        self.K = torch.tensor(
            [
                [focal, 0, width * 0.5],
                [0, focal, height * 0.5],
                [0, 0, 1],
            ]
        )

        self.K_inv = torch.inverse(self.K)

    def _sphere_intersection(self, ray_origin, ray_direction):
        # unit sphere
        center = torch.tensor([0, 0, 0])
        radius = torch.tensor(1 + self.sphere_eps)

        Q = ray_origin - center
        b = (ray_direction * (Q)) * 2
        b = b.sum(1)
        # print(b)

        # b = 2 * torch.matmul(ray_direction, ray_origin - center)
        c = torch.norm(Q, dim=1) ** 2 - radius**2
        # print(c)

        delta = b**2 - 4 * c
        # print(delta)

        t1 = (-b + delta.sqrt()) / 2
        t2 = (-b - delta.sqrt()) / 2

        step_to_sphere = torch.nan_to_num(torch.min(t1, t2))
        intersections = ray_origin + ray_direction * step_to_sphere.unsqueeze(1)
        intersection_mask = intersections.norm(dim=1) > (
            1 + (self.sphere_eps * 2)
        )  # fp inaccuracies
        return intersections, intersection_mask

    def _get_rays(self, pose):
        # pixel
        # TODO do this with K_inv
        # pixel_xs, pixel_ys = torch.meshgrid(
        #     torch.arange(self.width), torch.arange(self.height)
        # )
        # image_plane_pixel = torch.stack(
        #     [-pixel_xs, -pixel_ys, torch.ones_like(pixel_xs)], axis=-1
        # )

        # Screen coordinates
        pixel_xs, pixel_ys = torch.meshgrid(
            torch.arange(self.width), torch.arange(self.height)
        )
        # transform image -> camera
        # TODO do this with K_inv

        xs = (pixel_xs - self.width * 0.5) / self.focal
        ys = (pixel_ys - self.height * 0.5) / self.focal

        # homogeneous coordinates
        image_plane_camera = torch.stack(
            [-xs, -ys, torch.ones_like(pixel_xs), torch.ones_like(pixel_xs)], axis=-1
        )

        image_plane_world = pose @ image_plane_camera.view(-1, 4).T
        image_plane_world = image_plane_world.T.view(self.width, self.height, 4)
        image_plane_world_coord = image_plane_world[:, :, :3]

        self.plane_normal, self.c = self._get_image_plane(image_plane_world_coord)
        ray_origins = pose[:3, -1].expand(image_plane_world_coord.shape)

        return ray_origins, image_plane_world_coord - ray_origins

        # directions = torch.stack(
        #     [-xs, -ys, torch.ones_like(pixel_xs)], axis=-1
        # )  # (W,H,3)

        # transformed_dirs = directions.unsqueeze(2)  # 32x32x1x3
        # # transform camera to world
        # # TODO capture camera/extrinsic matrix in Camera .self
        # camera_dirs = transformed_dirs * R  # 32x32x3
        # ray_directions = camera_dirs.sum(axis=-1)  # 32x32x3

        # ray_origins = T.expand(ray_directions.shape)
        # # TODO calculate normals by transforming point into camera coordiate system and extracting the z coordinate
        # self.plane_normal, self.c = self._get_image_plane(ray_directions + ray_origins)
        # return ray_origins, ray_directions

    # def precompute_intersection(self, pose):
    #     ray_origin, ray_direction = self._get_rays(pose)
    #     ray_origin, ray_direction = ray_origin.reshape(-1, 3), ray_direction.reshape(
    #         -1, 3
    #     )
    #     ray_direction = ray_direction / torch.norm(ray_direction, dim=-1, keepdim=True)
    #     sphere_intersect_rays, sphere_mask = self._sphere_intersection(
    #         ray_origin, ray_direction
    #     )
    #     return sphere_intersect_rays, sphere_mask, ray_direction

    def render_depthmap(
        self, model: L.LightningModule, pose: torch.tensor, device="cpu"
    ):
        ray_origin, ray_direction = self._get_rays(pose)
        ray_origin, ray_direction = ray_origin.reshape(-1, 3), ray_direction.reshape(
            -1, 3
        )
        ray_direction = ray_direction / torch.norm(ray_direction, dim=-1, keepdim=True)
        sphere_intersect_rays, sphere_mask = self._sphere_intersection(
            ray_origin, ray_direction
        )

        location = self._find_surface(
            model, sphere_intersect_rays, ray_direction, sphere_mask, device=device
        )
        mask = location.norm(dim=1) > (1 + self.sphere_eps * 2)
        # TODO
        # [x] understand where the image plane lies
        # [x] calculate distance from point to image plane
        # [ ] calculation to be done on GPU
        depth = self._calc_distance_to_plane(location).cpu()

        depth[mask] = 0
        self.depth = depth.view(self.height, self.width).T

    def render_normal(
        self,
        model: L.LightningModule,
        pose,
        device="cuda",
    ):
        # TODO remove all CPU operations that can be precomputee
        ray_origin, ray_direction = self._get_rays(pose)
        ray_origin, ray_direction = ray_origin.reshape(-1, 3), ray_direction.reshape(
            -1, 3
        )
        ray_direction = ray_direction / torch.norm(ray_direction, dim=-1, keepdim=True)
        sphere_intersect_rays, sphere_mask = self._sphere_intersection(
            ray_origin, ray_direction
        )

        location, normals = self._find_surface(
            model,
            sphere_intersect_rays,
            ray_direction,
            sphere_mask,
            calc_normals=True,
            device=device,
        )
        mask = location.norm(dim=1) > (1 + self.sphere_eps * 2)

        normals = normalize(normals)

        normals = normals.cpu()
        normals[mask] = torch.tensor([1, 1, 1]).float()
        normals = normals.view(self.height, self.width, 3).transpose(0, 1)
        self.normal = (((normals + 1) / 2) * 255).numpy().astype("uint8")

    # def render_normal_precomputed(
    #     self,
    #     model: L.LightningModule,
    #     ray_origin: torch.tensor,
    #     device="cuda",
    # ):
    #     # TODO remove all CPU operations that can be precomputee
    #     ray_origin, ray_direction = self._get_rays(pose)
    #     ray_origin, ray_direction = ray_origin.reshape(-1, 3), ray_direction.reshape(
    #         -1, 3
    #     )
    #     ray_direction = ray_direction / torch.norm(ray_direction, dim=-1, keepdim=True)
    #     sphere_intersect_rays, sphere_mask = self._sphere_intersection(
    #         ray_origin, ray_direction
    #     )

    #     location, normals = self._find_surface(
    #         model,
    #         sphere_intersect_rays,
    #         ray_direction,
    #         sphere_mask,
    #         calc_normals=True,
    #         device=device,
    #     )
    #     mask = location.norm(dim=1) > (1 + self.sphere_eps * 2)

    #     normals = normalize(normals)

    #     normals = normals.cpu()
    #     normals[mask] = torch.tensor([1, 1, 1]).float()
    #     normals = normals.view(self.height, self.width, 3).transpose(0, 1)
    #     self.normal = (((normals + 1) / 2) * 255).numpy().astype("uint8")

    def render_image(
        self,
        model: L.LightningModule,
        pose: torch.tensor,
        obj_reflection,
        device="cuda",
    ):
        # TODO verify that lightsource is not none
        # TODO add positility do render normal
        ray_origin, ray_direction = self._get_rays(pose)
        ray_origin, ray_direction = ray_origin.reshape(-1, 3), ray_direction.reshape(
            -1, 3
        )
        ray_direction = ray_direction / torch.norm(ray_direction, dim=-1, keepdim=True)
        sphere_intersect_rays, sphere_mask = self._sphere_intersection(
            ray_origin, ray_direction
        )

        location, normals = self._find_surface(
            model,
            sphere_intersect_rays,
            ray_direction,
            sphere_mask,
            calc_normals=True,
            device=device,
        )
        mask = location.norm(dim=1) > (1 + self.sphere_eps * 2)

        normals = normalize(normals)

        image = torch.ones_like(normals, device="cpu")
        image[~mask] = self._illuminate(
            obj_reflection, location[~mask], normals[~mask]
        ).cpu()

        self.image = image.view(self.height, self.width, 3).transpose(0, 1)

        # TODO calculate normals as in render_normal

        normals = normals.cpu()
        normals[mask] = torch.tensor([1, 1, 1]).float()
        normals = normals.view(self.height, self.width, 3).transpose(0, 1)
        self.normal = (((normals + 1) / 2) * 255).numpy().astype("uint8")

    def _illuminate(self, obj_reflection, surface, normal, device="cuda"):
        # TODO
        # [x] write a custome normalization fucntion
        # [ ] more elegant solution for .to(device)
        intersection_to_light = normalize(
            self.lightsource.position.to(device) - surface
        )
        normal_to_surface = normal.squeeze()
        camera = torch.tensor([0, 0, -4], device=device)

        # RGB
        illumination = torch.zeros_like(normal_to_surface, device=device)

        # ambient
        illumination += obj_reflection.ambient.to(device) * self.lightsource.ambient.to(
            device
        )

        # diffuse
        illumination += (
            obj_reflection.diffuse.to(device)
            * self.lightsource.diffuse.to(device)
            * (intersection_to_light * normal_to_surface).sum(dim=-1, keepdim=True)
        )

        # specular
        intersection_to_camera = normalize(camera - surface)
        H = normalize(intersection_to_light + intersection_to_camera)
        illumination += (
            obj_reflection.specular.to(device)
            * self.lightsource.specular.to(device)
            * (normal_to_surface * H).sum(dim=-1, keepdim=True)
            ** (obj_reflection.shininess / 4)
        )

        illumination = torch.clip(illumination, 0, 1)

        return illumination

    def _find_surface(
        self,
        model,
        origin,
        direction,
        surface_mask,
        n_steps=100,
        calc_normals=False,
        device="cuda",
    ):
        # if mask is true, then the point has already reached his destination
        device = torch.device(device)
        location = origin.unsqueeze(0).to(device)
        surface = torch.zeros_like(location, device=device)
        surface_mask.to(device)
        sdfs = torch.zeros(location.shape[1], device=device)
        model.to(device)
        model.eval()
        model.latent = model.latent.to(device)
        direction = direction.to(device)
        for i in range(n_steps):
            # TODO introduce masks and debug
            # [x] introduce mask
            # [x] torch no grad()? (.predict())
            # [x] make 1e-3 a hyperparameter
            # [x] mask the points that don't intersect with unit sphere
            # [ ] remove all the squeeze and unsqueeze and check whether it improves performacne
            # [ ] visualize for different step sizes
            # [x] mask points that overshooted
            # [ ] custom function for the .to(device) calls
            sdf = model.predict(location[:, ~surface_mask.squeeze()])
            sdfs[~surface_mask.squeeze()] = sdf.squeeze()
            surface_mask = (sdfs < self.surface_eps) | (
                location.norm(dim=2) > (1 + self.sphere_eps * 2)
            )
            inv_acceleration = min(i / 10, 1)
            location[:, ~surface_mask.squeeze()] = (
                location + direction * sdfs.unsqueeze(1) * inv_acceleration
            )[:, ~surface_mask.squeeze()]

        surface[:, surface_mask.squeeze()] = location[:, surface_mask.squeeze()]

        if calc_normals:
            actual_surface_mask = location.norm(dim=2) > (1 + self.sphere_eps * 2)
            normals = torch.ones_like(location)
            inp = location[:, ~actual_surface_mask.squeeze()]
            inp.requires_grad_()
            out = model(inp)
            normals[~actual_surface_mask] = torch.autograd.grad(
                outputs=out,
                inputs=inp,
                grad_outputs=torch.ones_like(out),
            )[0]
            return surface.squeeze(), normals.squeeze()
        else:
            return surface.squeeze()

    def _get_image_plane(self, points):
        points = points.view(-1, 3)
        c = points.mean(0)
        c_points = points - c
        # Use SVD to find the principal components
        _, _, Vt = torch.linalg.svd(c_points[:3])  # we dont need to use all points

        # The normal vector is the last row of the Vt matrix
        plane_normal = Vt[-1, :]

        return plane_normal, c

    def _calc_distance_to_plane(self, point_3d):
        device = point_3d.device
        vec_to_point = point_3d - self.c.to(device)
        orthogonal_component = vec_to_point * self.plane_normal.to(device)
        distance_to_plane = torch.norm(orthogonal_component, dim=1)
        return distance_to_plane

    def show_image(self):
        plt.imshow(self.image)

    def save_image(self, path):
        plt.imsave(path, self.image)

    def show_depth(self):
        im = plt.imshow(self.depth)
        plt.colorbar()
        plt.show()

    def save_depth(self, path):
        plt.imsave(path, self.depth)

    def show_normal(self):
        plt.imshow(self.normal)

    def save_normal(self, path):
        plt.imsave(path, self.normal)
