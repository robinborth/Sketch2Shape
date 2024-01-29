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

########################################################################
# Experiments
########################################################################

train_siamese:
	python scripts/train_siamese.py +experiment/train_siamese=shapenet_chair_4096

eval_siamese:
	python scripts/eval_siamese.py +experiment/eval_siamese=resnet18
	python scripts/eval_siamese.py +experiment/eval_siamese=clip
	python scripts/eval_siamese.py +experiment/eval_siamese=siamese

train_deepsdf:
	python scripts/train_deepsdf.py +experiment/train_deepsdf=shapenet_chair_4096

traverse_latent:
	python scripts/traverse_latent.py +experiment/traverse_latent=mean_train_1
	python scripts/traverse_latent.py +experiment/traverse_latent=mean_train_2
	python scripts/traverse_latent.py +experiment/traverse_latent=random_1
	python scripts/traverse_latent.py +experiment/traverse_latent=random_2
	python scripts/traverse_latent.py +experiment/traverse_latent=train_train_1
	python scripts/traverse_latent.py +experiment/traverse_latent=train_train_2

optimize_deepsdf:
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_train
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_val
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=train_mesh_ckpt_1000
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=train_mesh_ckpt_2000
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=train_mesh_ckpt_3000
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_val_ckpt_500
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_val_ckpt_1000
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_val_ckpt_1500
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_val_ckpt_2000
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_val_ckpt_2500

optimize_normals:
	python scripts/optimize_normals.py +experiment/optimize_normals=mean_val

optimize_sketch:
	python scripts/optimize_sketch.py +experiment/optimize_sketch=mean_val

optimize_chair:
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=chair_train_prior
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=chair_train_prior_close
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=chair_train_mean
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=chair_train_random

	python scripts/optimize_normals.py +experiment/optimize_normals=chair_train_prior
	python scripts/optimize_normals.py +experiment/optimize_normals=chair_train_prior_close
	python scripts/optimize_normals.py +experiment/optimize_normals=chair_train_mean
	python scripts/optimize_normals.py +experiment/optimize_normals=chair_train_random

	python scripts/optimize_sketch.py +experiment/optimize_sketch=chair_train_prior
	python scripts/optimize_sketch.py +experiment/optimize_sketch=chair_train_prior_close
	python scripts/optimize_sketch.py +experiment/optimize_sketch=chair_train_mean
	python scripts/optimize_sketch.py +experiment/optimize_sketch=chair_train_random