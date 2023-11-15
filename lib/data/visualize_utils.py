import numpy as np
import open3d as o3d

from lib.data.sdf_utils import scale_to_unit_sphere_o3d


def visualize_pointcloud(points: np.ndarray, sdf=None) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if sdf is not None:
        colors = np.zeros(points.shape)
        colors[sdf < 0, 2] = 1
        colors[sdf > 0, 0] = 1
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_plotly([pcd])


def visualize_obj(path: str, normalize=True) -> None:
    mesh = o3d.io.read_triangle_mesh(path)
    if normalize:
        mesh_ = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        mesh_ = scale_to_unit_sphere_o3d(mesh_)
        mesh = mesh_.to_legacy()
    o3d.visualization.draw_plotly([mesh])
