from dataclasses import dataclass, field

import cv2 as cv
import numpy as np
import open3d as o3d
import point_cloud_utils as pcu

from lib.data.metainfo import MetaInfo
from lib.render.mesh import render_normals, render_normals_everywhere

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

    def obj_ids_iter(self):
        if not self.skip:
            yield from self.metainfo.obj_ids
        for obj_id in self.metainfo.obj_ids:
            normals_dir = self.metainfo.normals_dir_path(obj_id=obj_id)
            sketches_dir = self.metainfo.sketches_dir_path(obj_id=obj_id)
            if not normals_dir.exists() or not sketches_dir.exists():
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
        normals = (normals + 1) / 2
        normals[~mask] = 1
        normals = normals.reshape(normals.shape[0], self.width, self.height, -1)
        normals = (normals * 255).astype(np.uint8)
        return normals

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
class PreprocessNormalEverywhere:
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
    delta: float = 5e-02
    n_steps: int = 10

    def __post_init__(self):
        self.metainfo = MetaInfo(data_dir=self.data_dir)

    def obj_ids_iter(self):
        yield from [
            "52310bca00e6a3671201d487ecde379e"
            "2948af0b6a12f1c7ad484915511ccff6"
            "92e2317fd0d0129bb910025244eec99a"
            "1459c329e2123d4fe5b03ab845ae95c"
        ]

    def normals_to_image(self, normals: np.ndarray):
        normals = (normals + 1) / 2
        normals = normals.reshape(normals.shape[0], self.width, self.height, -1)
        normals = (normals * 255).astype(np.uint8)
        return normals

    def preprocess(self, obj_id: str):
        mesh = self.metainfo.load_normalized_mesh(obj_id=obj_id)
        _, normals, _ = render_normals_everywhere(
            mesh=mesh,
            azims=self.azims,
            elevs=self.elevs,
            dist=self.dist,
            width=self.width,
            height=self.height,
            focal=self.focal,
            sphere_eps=self.sphere_eps,
            delta=self.delta,
            n_steps=self.n_steps,
        )
        normals_everywhere = self.normals_to_image(normals=normals)
        return normals_everywhere


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
