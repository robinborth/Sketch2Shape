from pathlib import Path

# from mesh_to_sdf import sample_sdf_near_surface
import mesh2sdf
import numpy as np
import open3d as o3d
import trimesh

# from lib.data.sketch import obj_path


def scale_to_unit_sphere_o3d(mesh: o3d.t.geometry.TriangleMesh):
    bb = mesh.get_axis_aligned_bounding_box()
    vertices = mesh.vertex.positions - bb.get_center()
    dist = np.linalg.norm(vertices.numpy(), axis=1)
    dist_max = np.max(dist)
    vertices /= dist_max
    mesh.vertex.positions = vertices
    return mesh


def scale_to_unit_sphere_trimesh(mesh: trimesh.Trimesh):
    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def sample_volume_unit_sphere(count: int) -> np.ndarray:
    """Sample points from unit sphere by rejection

    Args:
        count (int): number of points to sample

    Returns:
        np.ndarray: array of points, with n<=count samples
    """
    samples = np.random.uniform(low=-1, high=1, size=(int(count * 2.1), 3))
    norm = np.linalg.norm(samples, axis=1)
    samples = samples[norm <= 1]
    if samples.shape[0] > count:
        return samples[:count]
    else:
        return samples


def create_sdf_samples_grid(path: str, num_samples: int = 10000):
    # disable the warning
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # create mesh
    mesh = o3d.io.read_triangle_mesh(path)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # transform to unit sphere
    mesh = scale_to_unit_sphere_o3d(mesh)

    mesh = mesh.to_legacy()

    # sample & perturb points from mesh
    cloud = mesh.sample_points_uniformly(num_samples // 2)

    np_cloud = np.asarray(cloud.points)
    # np_normals = np.asarray(cloud.normalize_normals().normals)

    noise = np.random.normal(scale=np.sqrt(0.0025), size=(num_samples // 2, 3))
    noise_2 = np.random.normal(scale=np.sqrt(0.00025), size=(num_samples // 2, 3))

    # np_noised_normals_1 = np_normals * noise.reshape(-1, 1)
    # np_noised_normals_2 = np_normals * noise_2.reshape(-1, 1)

    np_noised_cloud = noise + np_cloud
    np_noised_cloud_2 = noise_2 + np_cloud

    # sample points form unit sphere
    points_sphere = sample_volume_unit_sphere(int(num_samples * 0.2))

    points = np.concatenate((np_noised_cloud, np_noised_cloud_2, points_sphere)).astype(
        np.float32
    )

    # retrieve the sdf values
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    signed_distance = scene.compute_signed_distance(points).numpy()
    return np.column_stack((points, signed_distance))


# def get_sdf_samples_(obj_id: Path, number_of_points=50000):
#     # import os

#     # os.environ["PYOPENGL_PLATFORM"] = "egl"

#     mesh = trimesh.load(obj_id, force="mesh")
#     points, sdf = sample_sdf_near_surface(mesh, number_of_points=number_of_points)

#     return np.column_stack((points, sdf))


def fix_mesh(path: str, mesh_scale=0.7, size=128):
    level = 2 / size
    mesh = trimesh.load(path, force="mesh")
    # normalize mesh
    # vertices = mesh.vertices
    # bbmin = vertices.min(0)
    # bbmax = vertices.max(0)
    # center = (bbmin + bbmax) * 0.5
    # scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    # vertices = (vertices - center) * scale

    # normalize mesh
    mesh = scale_to_unit_sphere_trimesh(mesh)

    # fix mesh
    sdf, mesh = mesh2sdf.compute(
        mesh.vertices, mesh.faces, size, fix=True, level=level, return_mesh=True
    )

    # output
    # mesh.vertices = mesh.vertices / scale + center
    path = path[:-4] + ".fixed.obj"
    mesh.export(path)
    return path


# def trimesh_to_o3d(mesh: trimesh.Trimesh):
