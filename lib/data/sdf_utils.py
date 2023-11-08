from pathlib import Path

import numpy as np
import open3d as o3d

# from lib.data.sketch import obj_path


def scale_to_unit_sphere(mesh: o3d.t.geometry.TriangleMesh):
    bb = mesh.get_axis_aligned_bounding_box()
    vertices = mesh.vertex.positions - bb.get_center()
    dist = np.linalg.norm(vertices.numpy(), axis=1)
    dist_max = np.max(dist)
    vertices /= dist_max
    mesh.vertex.positions = vertices
    return mesh


def create_sdf_samples_grid(path: str, num_samples: int = 10000):
    # disable the warning
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # create mesh
    mesh = o3d.io.read_triangle_mesh(path)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # transform to unit sphere
    mesh = scale_to_unit_sphere(mesh)

    mesh = mesh.to_legacy()

    # sample & perturb points from mesh
    cloud = mesh.sample_points_uniformly(num_samples // 2)

    np_cloud = np.asarray(cloud.points)
    np_normals = np.asarray(cloud.normalize_normals().normals)

    noise = np.random.normal(scale=np.sqrt(0.0025), size=np_cloud.shape[0])
    noise_2 = np.random.normal(scale=np.sqrt(0.00025), size=np_cloud.shape[0])

    np_noised_normals_1 = np_normals * noise.reshape(-1, 1)
    np_noised_normals_2 = np_normals * noise_2.reshape(-1, 1)

    np_noised_cloud = np_noised_normals_1 + np_cloud
    np_noised_cloud_2 = np_noised_normals_2 + np_cloud

    # sample points form unit sphere
    # TBD

    points = np.concatenate((np_noised_cloud, np_noised_cloud_2)).astype(np.float32)

    # retrieve the sdf values
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    signed_distance = scene.compute_signed_distance(points).numpy()
    return np.column_stack((points, signed_distance))


# def get_sdf_samples(obj_id: Path, number_of_points=50000):
#     # import os

#     # os.environ["PYOPENGL_PLATFORM"] = "egl"

#     mesh = trimesh.load(obj_id, force="mesh")
#     points, sdf = sample_sdf_near_surface(mesh, number_of_points=number_of_points)

#     return np.column_stack((points, sdf))
