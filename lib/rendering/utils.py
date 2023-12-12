import numpy as np


def elev_azim_to_unit_shere(elev: float = 0.0, azim: float = 0.0):
    x = np.sin(np.deg2rad(azim)) * np.cos(np.deg2rad(elev) * 2)
    y = np.sin(np.deg2rad(azim)) * np.sin(np.deg2rad(elev) * 2)
    z = np.cos(np.deg2rad(azim))
    return normalize(np.array([x, y, z]))


def normalize(point):
    return point / np.linalg.norm(point, axis=-1)[..., None]
