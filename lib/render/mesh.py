import numpy as np
import open3d as o3d

from lib.render.camera import Camera

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def render_normals(
    mesh: o3d.geometry.TriangleMesh,
    azims=[],
    elevs=[],
    dist: float = 4.0,
    width: int = 256,
    height: int = 256,
    focal: int = 512,
    sphere_eps: float = 1e-1,
):
    _mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(_mesh)

    _points = []
    _normals = []
    _masks = []
    for azim in azims:
        for elev in elevs:
            camera = Camera(
                azim=azim,
                elev=elev,
                dist=dist,
                width=width,
                height=height,
                focal=focal,
                sphere_eps=sphere_eps,
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

            # update the points
            points = points + t_hit[..., None] * rays

            _points.append(points)
            _normals.append(normals)
            _masks.append(mask)

    return np.stack(_points), np.stack(_normals), np.stack(_masks)
