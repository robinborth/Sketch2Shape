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

preprocessing:
	python scripts/copy_shapenet.py data=shapenet_chairs_sketch_small
	python scripts/create_sketches.py data=shapenet_chairs_sketch_small
	python scripts/copy_shapenet.py data=shapenet_chairs_sketch_medium
	python scripts/create_sketches.py data=shapenet_chairs_sketch_medium
	python scripts/copy_shapenet.py data=shapenet_chairs_sketch_large
	python scripts/create_sketches.py data=shapenet_chairs_sketch_large
	python scripts/copy_shapenet.py data=shapenet_chairs_sketch_xl_large
	python scripts/create_sketches.py data=shapenet_chairs_sketch_xl_large