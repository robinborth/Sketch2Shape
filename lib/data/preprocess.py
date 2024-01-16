from dataclasses import dataclass, field

import cv2 as cv
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree

from lib.data.metainfo import MetaInfo
from lib.render.mesh import render_normals

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class PreprocessMesh:
    def __init__(self, data_dir: str = "/data", skip: bool = True):
        self.skip = skip
        self.metainfo = MetaInfo(data_dir=data_dir)

    def obj_ids_iter(self):
        if not self.skip:
            yield from self.metainfo.obj_ids
        for obj_id in self.metainfo.obj_ids:
            path = self.metainfo.normalized_mesh_path(obj_id=obj_id)
            if not path.exists():
                yield obj_id

    def preprocess(self, obj_id: str):
        return self.metainfo.load_mesh(obj_id=obj_id, normalize=True)


@dataclass
class PreprocessSiamese:
    # processing settings
    data_dir: str = "/data"
    skip: bool = True
    # camera settings
    azims: list[int] = field(default_factory=list)
    elevs: list[int] = field(default_factory=list)
    dist: float = 4.0
    width: int = 256
    height: int = 256
    focal: int = 512
    sphere_eps: float = 1e-1
    # sketch settings
    t_lower: int = 100
    t_upper: int = 150
    aperture_size: int = 3  # 3, 5, 7
    l2_gradient: bool = True

    def __post_init__(self):
        self.metainfo = MetaInfo(data_dir=self.data_dir)

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
        normals = (normals + 1) / 2
        normals[~mask] = 1
        normals = normals.reshape(normals.shape[0], self.width, self.height, -1)
        normals = (normals * 255).astype(np.uint8)
        return normals

    def obj_ids_iter(self):
        if not self.skip:
            yield from self.metainfo.obj_ids
        for obj_id in self.metainfo.obj_ids:
            normals_dir = self.metainfo.normals_dir_path(obj_id=obj_id)
            sketches_dir = self.metainfo.sketches_dir_path(obj_id=obj_id)
            if not normals_dir.exists() or not sketches_dir.exists():
                yield obj_id

    def preprocess(self, obj_id: str):
        mesh = self.metainfo.load_normalized_mesh(obj_id=obj_id)
        _, normals, masks = render_normals(
            mesh=mesh,
            azims=self.azims,
            elevs=self.elevs,
            dist=self.dist,
            width=self.width,
            height=self.height,
            focal=self.focal,
            sphere_eps=self.sphere_eps,
        )
        normals = self.normals_to_image(normals=normals, mask=masks)
        sketches = self.image_to_sketch(normals)
        return normals, sketches


@dataclass
class PreprocessSDF:
    data_dir: str = "/data"
    skip: bool = True
    # camera settings
    azims: list[int] = field(default_factory=list)
    elevs: list[int] = field(default_factory=list)
    dist: float = 4.0
    width: int = 256
    height: int = 256
    focal: int = 512
    sphere_eps: float = 1e-1
    # base samples
    cloud_samples: int = 500000
    # sample surface points
    surface_samples: int = 50000
    # sample inside points
    inside_samples: int = 150000
    inside_buffer_multiplier: float = 1.2
    inside_delta_mean: float = 0.0
    inside_delta_var: float = 1e-02
    # sample outside points
    outside_samples: int = 250000
    outside_buffer_multiplier: float = 1.2
    outside_delta_mean: float = 0.0
    outside_delta_var: float = 5e-02
    # sample unit sphere points
    unit_samples: int = 50000
    unit_buffer_multiplier: float = 1.2
    unit_alpha: float = 2.0
    unit_beta: float = 0.5

    def __post_init__(self):
        self.metainfo = MetaInfo(self.data_dir)

    def _get_cloud_samples(self, num_samples: int):
        mask = np.random.choice(self.cloud_points.shape[0], num_samples)
        return self.cloud_points[mask], self.cloud_normals[mask]

    def _get_sdfs(self, query: np.ndarray):
        sdfs, idx = self.tree.query(query, k=1)
        nearest_point = self.cloud_points[idx.squeeze()]
        nearest_normal = self.cloud_normals[idx.squeeze()]
        inside_mask = np.sum((query - nearest_point) * nearest_normal, axis=-1) < 0
        sdfs[inside_mask] = -sdfs[inside_mask]
        return sdfs.squeeze()

    def _sample_surface(self, num_samples: int):
        points, _ = self._get_cloud_samples(num_samples)
        sdfs = np.zeros(points.shape[0])
        return points, sdfs

    def _sample_near_surface(
        self,
        num_samples: int = 10000,
        inside_surface: bool = True,
        buffer_multiplier: float = 1.0,
        delta_mean: float = 0.0,
        delta_var: float = 5e-02,
    ):
        assert buffer_multiplier >= 1.0
        num_query_samples = int(num_samples * buffer_multiplier)
        delta = np.abs(np.random.normal(delta_mean, delta_var, size=num_query_samples))
        if inside_surface:
            delta = -delta

        points, normals = self._get_cloud_samples(num_query_samples)
        points = points + normals * delta[..., None]
        sdfs = self._get_sdfs(query=points)

        valid_mask = sdfs < 0 if inside_surface else sdfs > 0
        valid_mask &= np.linalg.norm(points, axis=-1) <= 1.0

        points, sdfs = points[valid_mask], sdfs[valid_mask]

        # ensures that you have the required points and sdfs with recursion
        if sdfs.shape[0] < num_samples:
            print("WARNING: buffer_multiplier to low. TODO change to logger!")
            points, sdfs = self._sample_near_surface(
                num_samples=num_samples,
                inside_surface=inside_surface,
                buffer_multiplier=buffer_multiplier * 2,
                delta_mean=delta_mean,
                delta_var=delta_var,
            )

        final_mask = np.random.choice(points.shape[0], num_samples)
        return points[final_mask], sdfs[final_mask]

    def _sample_unit_ball(
        self,
        num_samples: int = 100000,
        buffer_multiplier: float = 1.2,
        alpha: float = 2.0,
        beta: float = 0.5,
    ):
        assert buffer_multiplier >= 1.0
        num_query_samples = int(num_samples * buffer_multiplier)
        circle = o3d.geometry.TriangleMesh.create_sphere()
        point_cloud = circle.sample_points_uniformly(number_of_points=num_query_samples)

        points = np.asarray(point_cloud.points)
        points = points / np.linalg.norm(points, axis=-1)[..., None]

        radius = np.random.beta(alpha, beta, points.shape[0])
        points = points * radius[..., None]

        sdfs = self._get_sdfs(points)
        outside_mask = sdfs > 0
        points, sdfs = points[outside_mask], sdfs[outside_mask]

        if points.shape[0] < num_samples:
            print("WARNING: buffer_multiplier to low. TODO change to logger!")
            points, sdfs = self._sample_unit_ball(
                num_samples=num_samples,
                buffer_multiplier=buffer_multiplier * 2,
                alpha=alpha,
                beta=beta,
            )

        final_mask = np.random.choice(points.shape[0], num_samples)
        return points[final_mask], sdfs[final_mask]

    def sample_surface(self):
        return self._sample_surface(num_samples=self.surface_samples)

    def sample_inside_surface(self):
        return self._sample_near_surface(
            num_samples=self.inside_samples,
            inside_surface=True,
            buffer_multiplier=self.inside_buffer_multiplier,
            delta_mean=self.inside_delta_mean,
            delta_var=self.inside_delta_var,
        )

    def sample_outside_surface(self):
        return self._sample_near_surface(
            num_samples=self.outside_samples,
            inside_surface=False,
            buffer_multiplier=self.outside_buffer_multiplier,
            delta_mean=self.outside_delta_mean,
            delta_var=self.outside_delta_var,
        )

    def sample_unit_ball(self):
        return self._sample_unit_ball(
            num_samples=self.unit_samples,
            buffer_multiplier=self.unit_buffer_multiplier,
            alpha=self.unit_alpha,
            beta=self.unit_beta,
        )

    def obj_ids_iter(self):
        if not self.skip:
            yield from self.metainfo.obj_ids
        for obj_id in self.metainfo.obj_ids:
            surface_samples_path = self.metainfo.surface_samples_path(obj_id=obj_id)
            sdf_samples_path = self.metainfo.sdf_samples_path(obj_id=obj_id)
            if not surface_samples_path.exists() or not sdf_samples_path.exists():
                yield obj_id

    def preprocess(self, obj_id: str):
        # update state of the sdf
        mesh = self.metainfo.load_normalized_mesh(obj_id)
        points, normals, masks = render_normals(
            mesh=mesh,
            azims=self.azims,
            elevs=self.elevs,
        )
        self.cloud_points, self.cloud_normals = points[masks], normals[masks]
        self.tree = KDTree(self.cloud_points, leaf_size=2)

        # sample points with sdf values
        unit_points, unit_sdfs = self.sample_unit_ball()
        inside_points, inside_sdfs = self.sample_inside_surface()
        outside_points, outside_sdfs = self.sample_outside_surface()
        # combine the samples together
        points = np.concatenate([unit_points, inside_points, outside_points])
        sdfs = np.concatenate([unit_sdfs, inside_sdfs, outside_sdfs])
        sdf_samples = np.concatenate([points, sdfs[..., None]], axis=-1)

        # sample points on the surface for evaluation
        surface_samples, _ = self.sample_surface()
        return sdf_samples, surface_samples
