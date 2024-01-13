import matplotlib.pyplot as plt
import numpy as np
import torch


def get_translation(t):
    mat = np.identity(4)
    mat[2][3] += t
    return mat


def R_x(theta: float):
    _theta = np.deg2rad(np.array(theta, dtype=np.float32))
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(_theta), -np.sin(_theta)],
            [0, np.sin(_theta), np.cos(_theta)],
        ]
    )


def R_y(theta: float):
    _theta = np.deg2rad(np.array(theta, dtype=np.float32))
    return np.array(
        [
            [np.cos(_theta), 0, np.sin(_theta)],
            [0, 1, 0],
            [-np.sin(_theta), 0, np.cos(_theta)],
        ]
    )


def R_z(theta: float):
    _theta = np.deg2rad(np.array(theta, dtype=np.float32))
    return np.array(
        [
            [np.cos(_theta), -np.sin(_theta), 0],
            [np.sin(_theta), np.cos(_theta), 0],
            [0, 0, 1],
        ]
    )


def R_azim_elev(azim: float = 0.0, elev: float = 0.0):
    return R_y(azim) @ R_x(elev)


def normalize(point):
    return point / np.linalg.norm(point, dim=-1)[..., None]


def dot(x, y):
    return (x * y).sum(dim=-1)[..., None]


def visualize_mask(camera, mask):
    mask_image = mask.view(camera.resolution, camera.resolution).detach().cpu().numpy()
    plt.imshow(mask_image.T)


def visualize_normals(normals):
    normals_image = (normals + 1) / 2
    normals_image[~normals.to(torch.bool)] = 0.0
    plt.imshow(normals_image.detach().cpu().numpy())
    plt.show()


def visualize_image(image):
    plt.imshow(image.detach().cpu().numpy())
    plt.show()
