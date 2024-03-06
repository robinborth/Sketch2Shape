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
# Demo 
########################################################################

streamlit:
	python -m streamlit run lib/demo/app.py

########################################################################
# Experiments
########################################################################

train_deepsdf:
	python scripts/train_deepsdf.py +experiment/train_deepsdf=shapenet_chair_4096

train_latent_loss:
	python scripts/train_loss.py +experiment/train_loss=latent_traverse

train_triplet_loss:
	python scripts/train_loss.py +experiment/train_loss=triplet_traverse

########################################################################
# Report Abblations
########################################################################

abblation_train_loss:
	python scripts/train_loss.py +experiment/train_loss=latent_synthetic
	python scripts/train_loss.py +experiment/train_loss=latent_rendered
	python scripts/train_loss.py +experiment/train_loss=latent_traverse
	python scripts/train_loss.py +experiment/train_loss=triplet_traverse

eval_deepsdf:
	python scripts/optimize_deepsdf.py +experiment/eval_deepsdf=triplet_traverse
	python scripts/optimize_deepsdf.py +experiment/eval_deepsdf=latent_traverse
	python scripts/optimize_deepsdf.py +experiment/eval_deepsdf=latent_rendered
	python scripts/optimize_deepsdf.py +experiment/eval_deepsdf=latent_synthetic 

eval_optimize:
	python scripts/optimize_sketch.py +experiment/optimize_sketch=silhouette
	python scripts/optimize_sketch.py +experiment/optimize_sketch=global
	python scripts/optimize_sketch.py +experiment/optimize_sketch=full


########################################################################
# Report 
########################################################################

main_demo:
	python scripts/create_video.py +experiment/create_video=main_demo

multi_view:
	python scripts/optimize_sketch.py +experiment/create_video=multi_view_demo

########################################################################
# Debug Abblations
########################################################################

eval_loss:
	python scripts/eval_loss.py +experiment/eval_loss=resnet18
	python scripts/eval_loss.py +experiment/eval_loss=clip
	python scripts/eval_loss.py +experiment/eval_loss=triplet_loss

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

optimize_deepsdf_couch:
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=couch_train_prior
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=couch_train_prior_close
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=couch_train_mean
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=couch_train_random

optimize_normals_couch:
	python scripts/optimize_normals.py +experiment/optimize_normals=couch_train_prior
	python scripts/optimize_normals.py +experiment/optimize_normals=couch_train_prior_close
	python scripts/optimize_normals.py +experiment/optimize_normals=couch_train_mean
	python scripts/optimize_normals.py +experiment/optimize_normals=couch_train_random

