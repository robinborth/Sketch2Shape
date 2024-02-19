import numpy as np
import open3d as o3d

from lib.visualize.image import image_grid

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def visualize_object(obj) -> None:
    o3d.visualization.draw_plotly([obj], up=[0, 1, 0], front=[0.1, 0.1, -0.1])


def visualize_pointcloud(points: np.ndarray, sdf=None) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if sdf is not None:
        colors = np.zeros(points.shape)
        colors[sdf > 0, 0] = 1
        colors[sdf < 0, 2] = 1
        pcd.colors = o3d.utility.Vector3dVector(colors)
    visualize_object(pcd)


def visualize_sdf_slice(
    mesh: o3d.geometry.TriangleMesh,
    dim: str = "x",
    mask: bool = False,
):
    scene = o3d.t.geometry.RaycastingScene()
    _mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    _ = scene.add_triangles(_mesh)

    min_bound = _mesh.vertex.positions.min(0).numpy()
    max_bound = _mesh.vertex.positions.max(0).numpy()

    N = 256
    query_points = np.random.uniform(
        low=min_bound,
        high=max_bound,
        size=[N, 3],
    ).astype(np.float32)

    # Compute the signed distance for N random points
    signed_distance = scene.compute_signed_distance(query_points)
    xyz_range = np.linspace(max_bound, min_bound, num=32)

    # query_points is a [32,32,32,3] array ..
    query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32)
    query_points = np.swapaxes(query_points, 0, 1)

    # signed distance is a [32,32,32] array
    signed_distance = scene.compute_signed_distance(query_points)
    if mask:
        signed_distance = signed_distance < 0

    # We can visualize a slice of the distance field directly with matplotlib
    slices = []
    for idx in range(32):
        if dim == "x":
            slices.append(signed_distance.numpy()[idx, :, :])
        if dim == "y":
            slices.append(signed_distance.numpy()[:, idx, :])
        if dim == "z":
            slices.append(signed_distance.numpy()[:, :, idx])
    slices = np.stack(slices)
    return image_grid(slices, rows=8, cols=4)
