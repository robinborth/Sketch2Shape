# Sketch2Shape
The project for the computer vision lab. 

## Requirenments 

- Ubuntu 20.04 LTS
- Python 3.10
- Pytorch 2.1

For further information how to install the requirenemtns see the section in installation.

## Installation

In order to install mini-conda for python installation on ubuntu execture the following [https://docs.conda.io/projects/miniconda/en/latest/](ref):
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

In order to disable that the base environment is set:

```bash
conda config --set auto_activate_base false
```

Then you can create an environment for python3.11 like that:

```bash
conda create --name sketch2shape python=3.10
```

Then install the requirements.txt like following:
```bash
conda activate sketch2shape
pip install -r requirements.txt
pip install -e .
```

In order to find the interpreter path for VSCode do following:
```bash
conda activate sketch2shape
which python
```

## CMake & Open3D
In order to utilzie headless rendering in open3d, we need to build open3d with OSMesa from source. Please follow the official [docs](http://www.open3d.org/docs/release/tutorial/visualization/headless_rendering.html). Further make sure that CMake 3.20 is installed, [here](https://vitux.com/how-to-install-cmake-on-ubuntu/) is an detailed explenation how to install it on Ubuntu3.20. Note that this can take some time in order to install into the python environment.

## Pytorch3D

In order to install pytorch3D we need to use conda, the full installation guid is [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

```bash
conda install -c pytorch3d pytorch3d
```

```

```

## Exectuing Skripts

In order to execture a skript e.g. running the training or pre-processing run the following command, note that
you need to be in the environment and have the local project installed

```bash
python scripts/shape2sketch.py
```

or use the makefile

```bash
make shape2sketch
```

for more information which kind of skripts can be exectured look into the Makefile.

## Dataset

In order to download ShapeNet you need to register and get approved from the official website [here](https://shapenet.org/)
or alternative for quick testing you can download the dataset from kaggle unofficially [here](https://www.kaggle.com/datasets/jeremy26/shapenet-core/)

## Open Tasks
- [] Download the official ShapeNet dataset
- [] Render an image from one pointcloud 
- [] Extract edges from one images
- [] Create a skript that creates an aligned dataset from shapes to sketches
- [] Convert PointCloud/Mesh to VoxelGrid (occupancy values)
- [] Train a simple AutoEncoder from Sketch2VoxelGrid overfit to a few instances
- [] Look at the instances via WandB but also per instance with open3d or similar tool
- [] Calculate some metric with that IuO if VoxelGrid with ground truth as super easy baseline