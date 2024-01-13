import matplotlib.pyplot as plt
import numpy as np
import torch

# TODO change to numpy


def get_translation(t):
    mat = np.identity(4)
    mat[2][3] += t
    return mat


def get_rotation_x(theta: float):
    _theta = np.deg2rad(np.array(theta, dtype=np.float32))
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(_theta), -np.sin(_theta), 0],
            [0, np.sin(_theta), np.cos(_theta), 0],
            [0, 0, 0, 1],
        ]
    )


def get_rotation_y(theta: float):
    _theta = np.deg2rad(np.array(theta, dtype=np.float32))
    return np.array(
        [
            [np.cos(_theta), 0, np.sin(_theta), 0],
            [0, 1, 0, 0],
            [-np.sin(_theta), 0, np.cos(_theta), 0],
            [0, 0, 0, 1],
        ]
    )


def get_rotation_z(theta: float):
    _theta = np.deg2rad(np.array(theta, dtype=np.float32))
    return np.array(
        [
            [np.cos(_theta), -np.sin(_theta), 0, 0],
            [np.sin(_theta), np.cos(_theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def visualize_mask(camera, mask):
    mask_image = mask.view(camera.resolution, camera.resolution).detach().cpu().numpy()
    plt.imshow(mask_image.T)


def visualize_image(image):
    image = image.detach().cpu().numpy()
    plt.imshow(image)
    plt.show()
    return image
