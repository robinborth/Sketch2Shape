# Sketch2Shape

The project for the computer vision lab.

## Requirenments

- Ubuntu 20.04 LTS
- Python 3.11
- Pytorch 2.1

## Installation

In order install the dependencies please execute the following.

```bash
conda create -n sketch2shape python=3.11
conda activate sketch2shape
pip install -r requirements.txt
pip install -e .
```

## Skripts

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
