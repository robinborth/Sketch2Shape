import matplotlib.pyplot as plt
import torch


def R_x(theta: float):
    _theta = torch.deg2rad(torch.tensor(theta, dtype=torch.float32))
    return torch.tensor(
        [
            [1, 0, 0],
            [0, torch.cos(_theta), -torch.sin(_theta)],
            [0, torch.sin(_theta), torch.cos(_theta)],
        ]
    )


def R_y(theta: float):
    _theta = torch.deg2rad(torch.tensor(theta, dtype=torch.float32))
    return torch.tensor(
        [
            [torch.cos(_theta), 0, torch.sin(_theta)],
            [0, 1, 0],
            [-torch.sin(_theta), 0, torch.cos(_theta)],
        ]
    )


def R_z(theta: float):
    _theta = torch.deg2rad(torch.tensor(theta, dtype=torch.float32))
    return torch.tensor(
        [
            [torch.cos(_theta), -torch.sin(_theta), 0],
            [torch.sin(_theta), torch.cos(_theta), 0],
            [0, 0, 1],
        ]
    )


def R_azim_elev(azim: float = 0.0, elev: float = 0.0):
    return R_y(azim) @ R_x(elev)


def normalize(point):
    return point / torch.linalg.norm(point, dim=-1)[..., None]


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


def visualize_depth(depth):
    plt.imshow(depth.detach().cpu().numpy())
    plt.show()


def visualize_image(image):
    plt.imshow(image.detach().cpu().numpy())
    plt.show()
