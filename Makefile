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
# Experiments
########################################################################

experiments:
	# python scripts/traverse_latent.py +experiment/traverse_latent=version_1
	# python scripts/traverse_latent.py +experiment/traverse_latent=version_2

	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=train_optimization
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=val_optimization

	# python scripts/eval_siamese.py +experiment/eval_siamese=shapenet_chair_1024_resnet18
	# python scripts/eval_siamese.py +experiment/eval_siamese=shapenet_chair_1024_clip

	# python scripts/train_siamese.py +experiment/train_siamese=shapenet_chair_1024