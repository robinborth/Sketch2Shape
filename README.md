# Sketch2Shape
The project for the computer vision lab. 

## Requirenments 

- Ubuntu 20.04 LTS
- Python 3.11
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
conda create --name sketch2shape python=3.11
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
