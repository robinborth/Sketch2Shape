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


def render_normals_everywhere(
    mesh: o3d.geometry.TriangleMesh,
    azims=[],
    elevs=[],
    dist: float = 4.0,
    width: int = 256,
    height: int = 256,
    focal: int = 512,
    sphere_eps: float = 1e-1,
    delta: float = 5e-2,
    n_steps: int = 50,
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
            # calculate normals for stuff that hits surface
            points, rays = camera.rays()

            # ray casting
            camera_rays = np.concatenate([points, rays], axis=-1)
            out = scene.cast_rays(camera_rays)

            # normals map
            t_hit = out["t_hit"].numpy()
            mask = t_hit != np.inf
            t_hit[~mask] = 0

            # correct normals
            normals = out["primitive_normals"].numpy()
            outside_mask = (normals * rays).sum(axis=-1) > 0
            normals[outside_mask] = -normals[outside_mask]

            # update the points
            points[mask] = (points + t_hit[..., None] * rays)[mask]

            # calculate normals for rays that dont intersect with surface
            min_dist = np.full((width, height), np.inf)
            min_dist_points = points.copy()

            # fast start
            step_size = 2 / n_steps
            points[~mask] += (rays * 3)[~mask]
            for _ in range(n_steps):
                points[~mask] += (rays * step_size)[~mask]

                # calculate distance
                dis = scene.compute_distance(points).numpy()
                dist_mask = min_dist > dis
                mmask = dist_mask & (~mask)

                min_dist[mmask] = dis[mmask]
                min_dist_points[mmask] = points[mmask]

            d1 = (min_dist_points + [delta, 0, 0]).astype(np.float32)
            d2 = (min_dist_points - [delta, 0, 0]).astype(np.float32)
            d3 = (min_dist_points + [0, delta, 0]).astype(np.float32)
            d4 = (min_dist_points - [0, delta, 0]).astype(np.float32)
            d5 = (min_dist_points + [0, 0, delta]).astype(np.float32)
            d6 = (min_dist_points - [0, 0, delta]).astype(np.float32)

            # calculate approximate normals
            dist1 = scene.compute_distance(d1).numpy()
            dist2 = scene.compute_distance(d2).numpy()
            dist3 = scene.compute_distance(d3).numpy()
            dist4 = scene.compute_distance(d4).numpy()
            dist5 = scene.compute_distance(d5).numpy()
            dist6 = scene.compute_distance(d6).numpy()

            normal_ev = np.stack([dist1 - dist2, dist3 - dist4, dist5 - dist6], axis=-1)

            normal_ev /= np.linalg.norm(normal_ev, axis=-1, keepdims=True)

            normals[~mask] = normal_ev[~mask]

            _points.append(points)
            _normals.append(normals)
            _masks.append(mask)

    return np.stack(_points), np.stack(_normals), np.stack(_masks)
