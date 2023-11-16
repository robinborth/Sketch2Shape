########################################################################
# # Utils
########################################################################

install:
	conda create -n sketch2shape python=3.11
	conda activate sketch2shape
	pip install -r requirements.txt
	pip install -e .


format:
	black lib
	isort lib
	mypy lib
	flake8 lib

clean:
	rm -rf outputs/
	rm -rf wandb/

########################################################################
# Data Preprocessing
########################################################################

train_siamese:
	python scripts/run_train.py

copy_shapenet:
	python scripts/copy_shapenet.py

create_sketches:
	python scripts/create_sketches.py



########################################################################
# Experiments
########################################################################

train_deepsdf:
	python scripts/train_deepsdf.py +experiment=deepsdf_overfit_scene logger=wandb
	python scripts/train_deepsdf.py +experiment=deepsdf_overfit_batch logger=wandb