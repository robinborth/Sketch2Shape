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

memory:
	memray run --live scripts/memory.py


preprocessing:
	python scripts/copy_shapenet.py data=shapenet_chairs_sketch_small
	python scripts/shape2sketch.py data=shapenet_chairs_sketch_small
	python scripts/copy_shapenet.py data=shapenet_chairs_sketch_medium
	python scripts/shape2sketch.py data=shapenet_chairs_sketch_medium
	python scripts/copy_shapenet.py data=shapenet_chairs_sketch_large
	python scripts/shape2sketch.py data=shapenet_chairs_sketch_large
	python scripts/copy_shapenet.py data=shapenet_chairs_sketch_xl_large
	python scripts/shape2sketch.py data=shapenet_chairs_sketch_xl_large


experiments_dataloading:
	python scripts/run_train.py +experiments=pre_load_pre_transform_small_1gb 
	python scripts/run_train.py +experiments=pre_load_pre_transform_medium_1gb 
	# python scripts/run_train.py +experiments=pre_load_pre_transform_large_1gb 
	# python scripts/run_train.py +experiments=pre_load_pre_transform_xl_large_1gb 

	python scripts/run_train.py +experiments=pre_load_dynamic_transform_small_1gb 
	python scripts/run_train.py +experiments=pre_load_dynamic_transform_medium_1gb 
	# python scripts/run_train.py +experiments=pre_load_dynamic_transform_large_1gb 
	# python scripts/run_train.py +experiments=pre_load_dynamic_transform_xl_large_1gb 

	python scripts/run_train.py +experiments=dynamic_load_dynamic_transform_small_1gb 
	python scripts/run_train.py +experiments=dynamic_load_dynamic_transform_medium_1gb 
	# python scripts/run_train.py +experiments=dynamic_load_dynamic_transform_large_1gb 
	# python scripts/run_train.py +experiments=dynamic_load_dynamic_transform_xl_large_1gb 