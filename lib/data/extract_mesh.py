# %%
import numpy as np
import open3d as o3d
import torch
from skimage.measure import marching_cubes

from lib.models.deepsdf import MLP, DeepSDF
from lib.utils import load_config

# def marching_cubes(model, checkpoint_path: str, save_path: str, N: int):
#     m = model.load_from_checkpoint(checkpoint_path)
#     grid_vals = torch.arange(-1, 1, float(1 / N))
#     grid = torch.meshgrid(grid_vals, grid_vals, grid_vals)

#     xyz = (
#         torch.stack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel()))
#         .transpose(1, 0)
#         .to(torch.float64)
#     )

#     # lat_vec = torch.zeros(xyz.shape[0]).int()
#     # lat_vec = deepsdf.lat_vecs(lat_vec)
#     # x = (xyz, lat_vec)
#     # sd = deepsdf.predict(xyz, lat_vec)
#     sd = deepsdf.predict(xyz)
#     sd_r = sd.reshape(2 * N, 2 * N, 2 * N).detach().numpy()

#     verts, faces, normals, values = marching_cubes(sd_r, level=0.0)

#     x_max = np.array([1, 1, 1])
#     x_min = np.array([-1, -1, -1])
#     verts = verts * ((x_max - x_min) / (2 * N)) + x_min
#     mesh = o3d.geometry.TriangleMesh()
#     mesh.vertices = o3d.utility.Vector3dVector(verts)
#     mesh.triangles = o3d.utility.Vector3iVector(faces)
#     o3d.io.write_triangle_mesh(save_path, mesh)


cfg = load_config()
npy_path = f"{cfg['data_path']}/03001627/3c4ed9c8f76c7a5ef51f77a6d7299806/points.npy"
N = 32
checkpoint_path = (
    "/Users/korth/TUM/09/NiessnerLab/sketch2shape/checkpoint/deepsdf/last-v8.ckpt"
)
# deepsdf = DeepSDF.load_from_checkpoint(checkpoint_path)
deepsdf = MLP.load_from_checkpoint(
    "/Users/korth/TUM/09/NiessnerLab/sketch2shape/checkpoint/deepsdf/last-v5.ckpt"
)

grid_vals = torch.arange(-1, 1, float(1 / N))
grid = torch.meshgrid(grid_vals, grid_vals, grid_vals)

xyz = (
    torch.stack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel()))
    .transpose(1, 0)
    .to(torch.float64)
)

# lat_vec = torch.zeros(xyz.shape[0]).int()
# lat_vec = deepsdf.lat_vecs(lat_vec)
# x = (xyz, lat_vec)
# sd = deepsdf.predict(xyz, lat_vec)
sd = deepsdf.predict(xyz)
sd -= 0.03
# %%
sd_r = sd.reshape(2 * N, 2 * N, 2 * N).detach().numpy()

verts, faces, normals, values = marching_cubes(sd_r, level=0.0)

x_max = np.array([1, 1, 1])
x_min = np.array([-1, -1, -1])
verts = verts * ((x_max - x_min) / (2 * N)) + x_min
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(verts)
mesh.triangles = o3d.utility.Vector3iVector(faces)
# o3d.io.write_triangle_mesh(f"{cfg['data_path']}/test.obj", mesh)
o3d.visualization.draw_plotly([mesh])
# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Display resulting triangular mesh using Matplotlib. This can also be done
# with mayavi (see skimage.measure.marching_cubes docstring).
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(verts[faces])
mesh.set_edgecolor("k")
ax.add_collection3d(mesh)

ax.set_xlabel("x-axis: a = 6 per ellipsoid")
ax.set_ylabel("y-axis: b = 10")
ax.set_zlabel("z-axis: c = 16")

ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
ax.set_ylim(0, 20)  # b = 10
ax.set_zlim(0, 32)  # c = 16

plt.tight_layout()
plt.show()

# %%
