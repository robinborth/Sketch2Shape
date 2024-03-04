from dataclasses import dataclass, field

import cv2 as cv
import numpy as np
import open3d as o3d
import pandas as pd
import point_cloud_utils as pcu
import torch

from lib.data.metainfo import MetaInfo
from lib.models.deepsdf import DeepSDF
from lib.render.camera import Camera

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


@dataclass
class PreprocessMesh:
    data_dir: str = "/data"
    skip: bool = True
    resolution: int = 20000
    smoothing: bool = True
    laplacian_num_iters: int = 2

    def __post_init__(self):
        self.metainfo = MetaInfo(data_dir=self.data_dir)

    def obj_ids_iter(self):
        if not self.skip:
            yield from self.metainfo.obj_ids
            return

        for obj_id in self.metainfo.obj_ids:
            path = self.metainfo.normalized_mesh_path(obj_id=obj_id)
            if not path.exists():
                yield obj_id

    def smooth_watertight_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        resolution: int = 20000,
        laplacian_num_iters: int = 2,
    ):
        # extract verts and faces
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        # watertight and smooth, note that it's not watertight
        vw, fw = pcu.make_mesh_watertight(verts, faces, resolution)
        if self.smoothing:
            vw = pcu.laplacian_smooth_mesh(
                vw,
                fw,
                num_iters=laplacian_num_iters,
                use_cotan_weights=True,
            )
        # translate back to open3d mesh
        verts = o3d.utility.Vector3dVector(vw)
        faces = o3d.utility.Vector3iVector(fw)
        mesh = o3d.geometry.TriangleMesh(verts, faces)
        mesh.compute_vertex_normals()
        return mesh

    def normalize_mesh(self, mesh: o3d.geometry.TriangleMesh):
        points = np.asarray(mesh.vertices)
        translate = (np.min(points, axis=0) + np.max(points, axis=0)) / 2.0
        points -= translate
        points /= np.linalg.norm(points, axis=-1).max()
        mesh.vertices = o3d.utility.Vector3dVector(points)
        return mesh

    def preprocess(self, obj_id: str):
        mesh = self.metainfo.load_mesh(obj_id=obj_id)
        sw_mesh = self.smooth_watertight_mesh(
            mesh,
            resolution=self.resolution,
            laplacian_num_iters=self.laplacian_num_iters,
        )
        return self.normalize_mesh(sw_mesh)


@dataclass
class PreprocessSynthetic:
    # processing settings
    data_dir: str = "/data"
    deepsdf_ckpt_path: str = "deepsdf.ckpt"
    skip: bool = True
    n_renderings: int = 10
    # camera settings
    random: bool = True
    azims: list[float] = field(default_factory=list)
    azim_std: float = 22.5
    elev: float = -20
    elev_std: float = 5
    dist: float = 4.0
    width: int = 256
    height: int = 256
    focal: int = 512
    sphere_eps: float = 1e-1
    # grayscale settings
    ambient: float = 0.2
    diffuse: float = 0.5
    # sketch settings
    t_lower: int = 100
    t_upper: int = 150
    aperture_size: int = 3  # 3, 5, 7
    l2_gradient: bool = True

    def __post_init__(self):
        self.metainfo = MetaInfo(data_dir=self.data_dir)
        self.deepsdf = DeepSDF.load_from_checkpoint(self.deepsdf_ckpt_path)
        self.deepsdf.freeze()
        self.deepsdf.eval()
        self.max_label = len(self.deepsdf.lat_vecs.weight)

    def obj_ids_iter(self):
        if not self.skip:
            yield from self.metainfo.obj_ids
            return

        for obj_id in self.metainfo.obj_ids:
            if self.random:
                config_path = self.metainfo.config_path(obj_id=obj_id, mode=0)
            else:
                config_path = self.metainfo.config_path(obj_id=obj_id, mode=9)
            if not config_path.exists():
                yield obj_id

    def image_to_sketch(self, images: np.ndarray):
        edges = []
        for image in images:
            edge = cv.Canny(
                image,
                self.t_lower,
                self.t_upper,
                L2gradient=self.l2_gradient,
            )
            edge = cv.bitwise_not(edge)
            edges.append(edge)
        edges = np.stack((np.stack(edges),) * 3, axis=-1)
        return edges

    def normals_to_image(self, normals: np.ndarray, mask: np.ndarray):
        normals[~mask] = 1.0
        normals = normals / np.linalg.norm(normals, axis=-1)[..., None]
        normals = (normals * 0.5) + 0.5
        normals[~mask] = 1.0
        normals = normals.reshape(normals.shape[0], self.width, self.height, -1)
        normals = (normals * 255).astype(np.uint8)
        return normals

    def render_normals(self, mesh: o3d.geometry.TriangleMesh):
        if self.random:
            elevs = [self.elev for _ in range(self.n_renderings)]
        else:
            elevs = self.elev  # type: ignore

        _mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(_mesh)

        _normals = []
        _masks = []
        _azims = []
        _elevs = []
        for azim in self.azims:
            for elev in elevs:
                if self.random:
                    random_azim = np.random.normal(azim, self.azim_std)
                    random_elev = np.random.normal(elev, self.elev_std)
                else:
                    random_azim = azim
                    random_elev = elev
                camera = Camera(
                    azim=random_azim,
                    elev=random_elev,
                    dist=self.dist,
                    width=self.width,
                    height=self.height,
                    focal=self.focal,
                    sphere_eps=self.sphere_eps,
                )
                points, rays, _ = camera.unit_sphere_intersection_rays()

                # sphere tracing
                camera_rays = np.concatenate([points, rays], axis=-1)
                out = scene.cast_rays(camera_rays)

                # noramls map
                t_hit = out["t_hit"].numpy()
                mask = t_hit != np.inf
                t_hit[~mask] = 0

                # correct normals
                normals = out["primitive_normals"].numpy()
                outside_mask = (normals * rays).sum(axis=-1) > 0
                normals[outside_mask] = -normals[outside_mask]

                _normals.append(normals)
                _masks.append(mask)
                _azims.append(random_azim)
                _elevs.append(random_elev)

        return np.stack(_normals), np.stack(_masks), _azims, _elevs

    def normals_to_grayscales(self, normals: np.ndarray, azims, elevs):
        """Transforms the normal after render_normals to grayscale."

        Args:
            normal (np.ndarray): The normal image of dim: (B, H, W, 3) and range (0, 1)

        Returns:
            np.ndarray: The grayscale image of dim (B, H, W, 3) and range (0, 1)
        """
        grayscales = []
        view_id = 0
        for azim, elev in zip(azims, elevs):
            normal = normals[view_id].reshape(self.width, self.height, -1)
            camera_position = Camera(azim=azim, elev=elev).camera_position()
            mask = normal.sum(-1) == 0
            L = camera_position / np.linalg.norm(camera_position)
            grayscale = np.zeros(normal.shape)
            grayscale += self.ambient
            grayscale += self.diffuse * (normal @ L)[..., None]
            grayscale[mask, :] = 1.0
            grayscale = np.clip(grayscale, 0, 1)
            grayscales.append(grayscale)
            view_id += 1
        return (np.stack(grayscales) * 255).astype(np.uint8)

    def preprocess(self, obj_id: str):
        mesh = self.metainfo.load_normalized_mesh(obj_id=obj_id)
        normals, masks, azims, elevs = self.render_normals(mesh=mesh)
        grayscales = self.normals_to_grayscales(normals, azims=azims, elevs=elevs)
        normals = self.normals_to_image(normals=normals, mask=masks)
        sketches = self.image_to_sketch(normals)

        label = self.metainfo.obj_id_to_label(obj_id=obj_id)
        latents = []
        configs = []
        if label < self.max_label:
            for idx, (azim, elev) in enumerate(zip(azims, elevs)):
                latent = self.deepsdf.lat_vecs.weight[label]
                latents.append(latent.detach().cpu().numpy())
                config = dict(
                    image_id=f"{idx:05}",
                    source_obj_id=obj_id,
                    target_obj_id=obj_id,
                    t=0.0,
                    azim=azim,
                    elev=elev,
                    dist=self.dist,
                    width=self.width,
                    height=self.height,
                    focal=self.focal,
                    sphere_eps=self.sphere_eps,
                    ambient=self.ambient,
                    diffuse=self.diffuse,
                )
                configs.append(config)
        latents = np.stack(latents)
        configs = pd.DataFrame(configs)
        return normals, sketches, grayscales, latents, configs


@dataclass
class PreprocessRenderings:
    # processing settings
    data_dir: str = "/data"
    deepsdf_ckpt_path: str = "deepsdf.ckpt"
    skip: bool = True
    # traversal settings
    traversal: bool = True
    n_renderings: int = 10
    t_mean: float = 0.25
    t_std: float = 0.1
    # camera settings
    azims: list[float] = field(default_factory=list)
    azim_std: float = 22.5
    elev: float = -20
    elev_std: float = 5
    dist: float = 4.0
    width: int = 256
    height: int = 256
    focal: int = 512
    sphere_eps: float = 1e-1
    # grayscale settings
    ambient: float = 0.2
    diffuse: float = 0.5
    # sketch settings
    t_lower: int = 100
    t_upper: int = 150
    aperture_size: int = 3  # 3, 5, 7
    l2_gradient: bool = True
    split: str = "train_latent"  # train_latent, val_latent

    def __post_init__(self):
        self.metainfo = MetaInfo(data_dir=self.data_dir, split=self.split)
        self.deepsdf = DeepSDF.load_from_checkpoint(self.deepsdf_ckpt_path)
        self.deepsdf.freeze()
        self.deepsdf.eval()

    def obj_ids_iter(self):
        if not self.skip:
            yield from self.metainfo.obj_ids
            return

        for obj_id in self.metainfo.obj_ids:
            if self.traversal:
                config_path = self.metainfo.config_path(obj_id=obj_id, mode=6)
            else:
                config_path = self.metainfo.config_path(obj_id=obj_id, mode=3)
            if not config_path.exists():
                yield obj_id

    def image_to_sketch(self, image: np.ndarray):
        edge = cv.Canny(
            image,
            self.t_lower,
            self.t_upper,
            L2gradient=self.l2_gradient,
        )
        edge = cv.bitwise_not(edge)
        return np.stack([edge] * 3, axis=-1)

    def preprocess(self, obj_id: str):
        source_label = self.metainfo.obj_id_to_label(obj_id)
        source_latent = self.deepsdf.lat_vecs.weight[source_label]
        target_idxs = [source_label] * self.n_renderings

        normals = []
        sketches = []
        grayscales = []
        latents = []
        configs = []

        for azim in self.azims:
            if self.traversal:
                permutation = np.random.permutation(len(self.metainfo.obj_ids))
                target_idxs = list(permutation[: self.n_renderings])
            for idx, target_idx in enumerate(target_idxs):
                # get the latent from the target obj_id
                if self.traversal:
                    target_obj_id = self.metainfo.obj_ids[target_idx]
                    target_label = self.metainfo.obj_id_to_label(target_obj_id)
                    target_latent = self.deepsdf.lat_vecs.weight[target_label]
                else:
                    target_obj_id = obj_id
                    target_latent = source_latent

                # set the camera settings
                rand_azim = np.random.normal(azim, self.azim_std)
                rand_elev = np.random.normal(self.elev, self.elev_std)
                camera_config = dict(
                    azim=rand_azim,
                    elev=rand_elev,
                    dist=self.dist,
                    width=self.width,
                    height=self.height,
                    focal=self.focal,
                    sphere_eps=self.sphere_eps,
                )
                self.deepsdf.create_camera(**camera_config)
                camera_config["ambient"] = self.ambient
                camera_config["diffuse"] = self.diffuse

                # interpolate between the source latent and the target
                t = torch.normal(torch.tensor(self.t_mean), torch.tensor(self.t_std))
                t = torch.clamp(t, 0.0, 0.5)
                if not self.traversal:  # disable traversal
                    t = torch.tensor(0.0)
                interpolated_latent = (1 - t) * source_latent + t * target_latent
                latents.append(interpolated_latent.detach().cpu().numpy())

                # render the normals and the sketch
                rendered_normal = self.deepsdf.capture_camera_frame(interpolated_latent)
                normal = (rendered_normal.detach().cpu().numpy() * 255).astype(np.uint8)
                sketch = self.image_to_sketch(normal)
                rendered_grayscale = self.deepsdf.normal_to_grayscale(
                    normal=rendered_normal,
                    ambient=self.ambient,
                    diffuse=self.diffuse,
                ).detach()
                grayscale = (rendered_grayscale.cpu().numpy() * 255).astype(np.uint8)

                # update the images
                grayscales.append(grayscale)
                normals.append(normal)
                sketches.append(sketch)
                # update the configs
                config = {
                    "image_id": f"{idx:05}",
                    "source_obj_id": obj_id,
                    "target_obj_id": target_obj_id,
                    "t": t.item(),
                }
                config.update(camera_config)
                configs.append(config)
        latents = np.stack(latents)
        configs = pd.DataFrame(configs)
        return normals, sketches, grayscales, latents, configs


@dataclass
class PreprocessSDF:
    data_dir: str = "/data"
    skip: bool = True
    # sample surface points
    surface_samples: int = 50000
    # sample near points
    near_samples_1: int = 100000
    near_scale_1: float = 5e-03
    near_buffer_1: float = 1.1
    # sample outside points
    near_samples_2: int = 100000
    near_scale_2: float = 5e-02
    near_buffer_2: float = 1.1
    # sample unit sphere points
    unit_samples: int = 100000
    unit_buffer: float = 2.0

    def __post_init__(self):
        self.metainfo = MetaInfo(self.data_dir)

    def obj_ids_iter(self):
        if not self.skip:
            yield from self.metainfo.obj_ids
            return

        for obj_id in self.metainfo.obj_ids:
            surface_samples_path = self.metainfo.surface_samples_path(obj_id=obj_id)
            sdf_samples_path = self.metainfo.sdf_samples_path(obj_id=obj_id)
            if not surface_samples_path.exists() or not sdf_samples_path.exists():
                yield obj_id

    def get_sdfs(self, query: np.ndarray):
        query_point = o3d.core.Tensor(query, dtype=o3d.core.Dtype.Float32)
        return self.scene.compute_signed_distance(query_point).numpy()

    def sample_surface(self, num_samples: int):
        pcd = self.mesh.sample_points_uniformly(num_samples)
        points = np.asarray(pcd.points)
        assert num_samples == points.shape[0]
        sdfs = np.zeros(points.shape[0])
        return points, sdfs

    def sample_near_surface(
        self,
        num_samples: int = 100000,
        scale: float = 5e-02,
        buffer: float = 1.1,
    ):
        total_points = 0
        tmp_points = []
        while total_points < num_samples:
            to_sample = int((num_samples - total_points) * buffer)
            surface_points, _ = self.sample_surface(to_sample)
            delta = np.random.normal(scale=scale, size=(to_sample, 3))
            points = surface_points + delta
            sphere_mask = np.linalg.norm(points, axis=-1) <= 1.0
            points = points[sphere_mask]  # (P, 3)
            tmp_points.append(points)
            total_points += len(points)
        points = np.concatenate(tmp_points)  # (N, 3)
        points = points[np.random.choice(points.shape[0], num_samples)]
        assert num_samples == points.shape[0]
        sdf = self.get_sdfs(points)
        return points, sdf

    def sample_unit_ball(self, num_samples: int = 100000, buffer: float = 2.0):
        total_points = 0
        tmp_points = []
        while total_points < num_samples:
            to_sample = int((num_samples - total_points) * buffer)
            points = np.random.uniform(-1, 1, size=(to_sample, 3))
            sphere_mask = np.linalg.norm(points, axis=-1) <= 1.0
            points = points[sphere_mask]
            tmp_points.append(points)
            total_points += len(points)
        points = np.concatenate(tmp_points)  # (N, 3)
        points = points[np.random.choice(points.shape[0], num_samples)]
        assert num_samples == points.shape[0]
        sdf = self.get_sdfs(points)
        return points, sdf

    def load(self, obj_id: str):
        self.mesh = self.metainfo.load_normalized_mesh(obj_id)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))

    def preprocess(self, obj_id: str):
        # extra function in order to load it outisde preprocess
        self.load(obj_id=obj_id)

        # sample points with sdf values
        unit_points, unit_sdfs = self.sample_unit_ball(
            num_samples=self.unit_samples,
            buffer=self.unit_buffer,
        )
        near_point_1, near_sdfs_1 = self.sample_near_surface(
            num_samples=self.near_samples_1,
            scale=self.near_scale_1,
            buffer=self.near_buffer_1,
        )
        near_point_2, near_sdfs_2 = self.sample_near_surface(
            num_samples=self.near_samples_2,
            scale=self.near_scale_2,
            buffer=self.near_buffer_2,
        )
        # combine the samples together
        points = np.concatenate([unit_points, near_point_1, near_point_2])
        sdfs = np.concatenate([unit_sdfs, near_sdfs_1, near_sdfs_2])
        sdf_samples = np.concatenate([points, sdfs[..., None]], axis=-1)

        # sample points on the surface for evaluation
        surface_samples, _ = self.sample_surface(num_samples=self.surface_samples)

        return sdf_samples, surface_samples


############################################################
# Deprecated Preprocessing
############################################################
class PreprocessSiamese:
    pass
