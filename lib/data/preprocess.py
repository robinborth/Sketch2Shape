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

    def get_sdfs(self, query: np.ndarray, k: int = 5):
        # use the distances from query to mesh, because this is more accurate then
        # query to surface points, and then filter the bad ones out
        query_point = o3d.core.Tensor(query, dtype=o3d.core.Dtype.Float32)
        ans = self.scene.compute_closest_points(query_point)
        nearest_points = ans["points"].numpy()  # (P, 3)
        sdfs = np.linalg.norm(nearest_points - query, axis=-1)  # (P,)

        # retrieving 5 instead of 1 does not make such a big difference, e.g. for
        # 60000 points from 1.78s -> 1.93s, but we need it in order to filter bad points
        # note that we make a query with the nearest points on the surface
        _, idx = self.tree.query(nearest_points, k=k)  # (P, K)

        top_k_nearest_point = self.cloud_points[idx]  # (P, K, 3)
        top_k_nearest_normal = self.cloud_normals[idx]  # (P, K, 3)

        # get the normal direction of the closest points to the query
        query_vec = query[:, None, :] - top_k_nearest_point
        raw_inside_mask = np.sum(query_vec * top_k_nearest_normal, axis=-1) < 0

        # only consider inside or outside if no mismatch between top-k nearest points
        inside_mask = raw_inside_mask.sum(-1) == k  # (P,)
        outside_mask = raw_inside_mask.sum(-1) == 0  # (P,)
        mask = inside_mask | outside_mask  # (P,)

        sdfs[inside_mask] = -sdfs[inside_mask]
        return sdfs.squeeze(), mask  # (N,)

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
        sdfs, mask = self.get_sdfs(query=points)

        valid_mask = sdfs < 0 if inside_surface else sdfs > 0
        valid_mask &= np.linalg.norm(points, axis=-1) <= 1.0
        valid_mask &= mask

        points, sdfs = points[valid_mask], sdfs[valid_mask]
        return points, sdfs
        # ensures that you have the required points and sdfs with recursion
        # if sdfs.shape[0] < num_samples:
        #     print("WARNING: buffer_multiplier to low. TODO change to logger!")
        #     points, sdfs = self._sample_near_surface(
        #         num_samples=num_samples,
        #         inside_surface=inside_surface,
        #         buffer_multiplier=buffer_multiplier * 2,
        #         delta_mean=delta_mean,
        #         delta_var=delta_var,
        #     )

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

        sdfs = self.get_sdfs(points)
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

    def load(self, obj_id: str):
        # update state of the sdf
        self.mesh = self.metainfo.load_normalized_mesh(obj_id)

        mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(mesh)

        # sample normals and points from the surface and create a KDTree this is used
        # to remove sdf values after find the closest point, because the mesh is not
        # watertight.
        pcl = self.mesh.sample_points_uniformly(self.cloud_samples)
        self.cloud_points = np.asarray(pcl.points)
        normals = np.asarray(pcl.normals)
        self.cloud_normals = normals / np.linalg.norm(normals, axis=-1)[..., None]
        self.tree = KDTree(self.cloud_points, leaf_size=2)

    def preprocess(self, obj_id: str):
        # extra function in order to load it outisde preprocess
        self.load(obj_id=obj_id)

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
