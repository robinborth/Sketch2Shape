from pathlib import Path

import numpy as np
import trimesh


def scale_to_unit_sphere(mesh: trimesh.Trimesh, scale, offset):
    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def scale_to_unit_cube(mesh: trimesh.Trimesh, scale, offset):
    centroid = np.mean(
        mesh.vertices, axis=0
    )  # this sounds more like the mean, but it is how ShapeNetV2 officially normalizes the shapes
    min = np.min(mesh.vertices, axis=0)
    max = np.max(mesh.vertices, axis=0)
    diag = max - min
    print(diag)
    norm = 1 / np.linalg.norm(diag, axis=0)
    vertices = (mesh.vertices - centroid) * norm

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
