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
