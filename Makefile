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

shape2sketch:
	python scripts/shape2sketch.py

train:
	python scripts/run_train.py

copy_shapenet:
	python scripts/copy_shapenet.py