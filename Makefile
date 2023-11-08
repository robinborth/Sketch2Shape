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
	python scripts/train_deepsdf.py

experiment_deep_sdf_batch_size:
	python scripts/train_deepsdf.py batch_size=128 tags=['deep_sdf,experiment:batch_size,batch_size:2048']
	# python scripts/train_deepsdf.py batch_size=1024 tags=['deep_sdf,experiment:batch_size,batch_size:1024']
	# python scripts/train_deepsdf.py batch_size=2048 tags=['deep_sdf,experiment:batch_size,batch_size:2048']
	# python scripts/train_deepsdf.py batch_size=5096 tags=['deep_sdf,experiment:batch_size,batch_size:5096']
	# python scripts/train_deepsdf.py batch_size=10192 tags=['deep_sdf,experiment:batch_size,batch_size:10192']