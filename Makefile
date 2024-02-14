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

train_deepsdf:
	python scripts/train_deepsdf.py +experiment/train_deepsdf=shapenet_chair_4096

train_loss:
	python scripts/train_loss.py +experiment/train_loss=siamese_shapenet_chair_4096

eval_loss:
	python scripts/eval_loss.py +experiment/eval_loss=resnet18
	python scripts/eval_loss.py +experiment/eval_loss=clip
	python scripts/eval_loss.py +experiment/eval_loss=siamese

traverse_latent:
	python scripts/traverse_latent.py +experiment/traverse_latent=mean_train_1
	python scripts/traverse_latent.py +experiment/traverse_latent=mean_train_2
	python scripts/traverse_latent.py +experiment/traverse_latent=mean_train_3
	python scripts/traverse_latent.py +experiment/traverse_latent=mean_train_4
	python scripts/traverse_latent.py +experiment/traverse_latent=random_1
	python scripts/traverse_latent.py +experiment/traverse_latent=random_2
	python scripts/traverse_latent.py +experiment/traverse_latent=random_3
	python scripts/traverse_latent.py +experiment/traverse_latent=random_4
	python scripts/traverse_latent.py +experiment/traverse_latent=train_train_1
	python scripts/traverse_latent.py +experiment/traverse_latent=train_train_2
	python scripts/traverse_latent.py +experiment/traverse_latent=train_train_3
	python scripts/traverse_latent.py +experiment/traverse_latent=train_train_4
	python scripts/traverse_latent.py +experiment/traverse_latent=siamese_train_train_1

optimize_deepsdf:
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=baseline
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_train
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_val
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=train_mesh_ckpt_1000
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=train_mesh_ckpt_2000
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_val_ckpt_500
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_val_ckpt_1000
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_val_ckpt_1500
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_val_ckpt_2000

optimize_normals:
	python scripts/optimize_normals.py +experiment/optimize_normals=mean_train

optimize_sketch:
	python scripts/optimize_sketch.py +experiment/optimize_sketch=optim_chair_prior

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

optimize_couch:
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=couch_train_prior
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=couch_train_prior_close
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=couch_train_mean
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=couch_train_random

	python scripts/optimize_normals.py +experiment/optimize_normals=couch_train_prior
	python scripts/optimize_normals.py +experiment/optimize_normals=couch_train_prior_close
	python scripts/optimize_normals.py +experiment/optimize_normals=couch_train_mean
	python scripts/optimize_normals.py +experiment/optimize_normals=couch_train_random

	python scripts/optimize_sketch.py +experiment/optimize_sketch=couch_train_prior
	python scripts/optimize_sketch.py +experiment/optimize_sketch=couch_train_prior_close
	python scripts/optimize_sketch.py +experiment/optimize_sketch=couch_train_mean
	python scripts/optimize_sketch.py +experiment/optimize_sketch=couch_train_random

optimize_couch_1:
	python scripts/optimize_sketch.py +experiment/optimize_sketch=optim_couch_1
	python scripts/optimize_sketch.py +experiment/optimize_sketch=optim_couch_1 model.reg_loss=True
	python scripts/optimize_sketch.py +experiment/optimize_sketch=optim_chair_1
	python scripts/optimize_sketch.py +experiment/optimize_sketch=optim_chair_1 model.reg_loss=True