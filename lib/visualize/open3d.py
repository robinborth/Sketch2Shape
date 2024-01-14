import numpy as np
import open3d as o3d

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
