import matplotlib.pyplot as plt
import torch


def get_translation(t):
    mat = torch.eye(4)
    mat[2][3] += t
    return mat


def get_rotation_x(deg):
    deg = torch.deg2rad(torch.tensor(deg))
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, torch.cos(deg), -torch.sin(deg), 0],
            [0, torch.sin(deg), torch.cos(deg), 0],
            [0, 0, 0, 1],
        ]
    )


def get_rotation_y(deg):
    deg = torch.deg2rad(torch.tensor(deg))
    return torch.tensor(
        [
            [torch.cos(deg), 0, torch.sin(deg), 0],
            [0, 1, 0, 0],
            [-torch.sin(deg), 0, torch.cos(deg), 0],
            [0, 0, 0, 1],
        ]
    )


def get_rotation_z(deg):
    deg = torch.deg2rad(torch.tensor(deg))
    return torch.tensor(
        [
            [torch.cos(deg), -torch.sin(deg), 0, 0],
            [torch.sin(deg), torch.cos(deg), 0, 0],
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
