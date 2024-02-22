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

train_latent_loss:
	python scripts/train_loss.py +experiment/train_loss=latent_siamese_multi_view

abblation_latent_loss:
	# python scripts/train_loss.py +experiment/train_loss=latent_siamese_edge_normal_multi_view_256
	# python scripts/train_loss.py +experiment/train_loss=latent_siamese_edge_grayscale_multi_view_256
	python scripts/train_loss.py +experiment/train_loss=latent_siamese_sketch_grayscale_multi_view_256

	# python scripts/train_loss.py +experiment/train_loss=latent_tower_edge_normal_multi_view_256
	# python scripts/train_loss.py +experiment/train_loss=latent_tower_edge_grayscale_multi_view_256
	python scripts/train_loss.py +experiment/train_loss=latent_tower_sketch_grayscale_multi_view_256

	python scripts/train_loss.py +experiment/train_loss=latent_siamese_sketch_grayscale_latent_256
	python scripts/train_loss.py +experiment/train_loss=latent_tower_sketch_grayscale_latent_256

	# python scripts/train_loss.py +experiment/train_loss=latent_siamese_sketch_grayscale_latent_multi_view_256
	# python scripts/train_loss.py +experiment/train_loss=latent_siamese_sketch_grayscale_multi_view_64
	# python scripts/train_loss.py +experiment/train_loss=latent_siamese_sketch_grayscale_multi_view_128

train_triplet_loss:
	python scripts/train_loss.py +experiment/train_loss=triplet_siamese_multi_view

train_barlow_loss:
	python scripts/train_loss.py +experiment/train_loss=triplet_siamese_multi_view

eval_loss:
	python scripts/eval_loss.py +experiment/eval_loss=resnet18
	python scripts/eval_loss.py +experiment/eval_loss=clip
	python scripts/eval_loss.py +experiment/eval_loss=triplet_loss

eval_latent_loss:
	python scripts/eval_loss.py +experiment/eval_loss=latent_loss loss_ckpt_path=checkpoints/latent_siamese_edge_grayscale_multi_view_256.ckpt
	python scripts/eval_loss.py +experiment/eval_loss=latent_loss loss_ckpt_path=checkpoints/latent_siamese_edge_grayscale_multi_view_256.ckpt +data/variants=eval_grayscale
	python scripts/eval_loss.py +experiment/eval_loss=latent_loss loss_ckpt_path=checkpoints/latent_siamese_edge_grayscale_multi_view_256.ckpt +data/variants=eval_sketch
	python scripts/eval_loss.py +experiment/eval_loss=latent_loss loss_ckpt_path=checkpoints/latent_siamese_edge_grayscale_multi_view_256.ckpt +data/variants=eval_grayscale +data/variants=eval_sketch

eval_latent_deepsdf:
	python scripts/eval_loss.py +experiment/eval_loss=latent_loss loss_ckpt_path=checkpoints/latent_siamese_edge_grayscale_multi_view_256.ckpt
	python scripts/eval_loss.py +experiment/eval_loss=latent_loss loss_ckpt_path=checkpoints/latent_siamese_edge_grayscale_multi_view_256.ckpt +data/variants=eval_grayscale
	python scripts/eval_loss.py +experiment/eval_loss=latent_loss loss_ckpt_path=checkpoints/latent_siamese_edge_grayscale_multi_view_256.ckpt +data/variants=eval_sketch
	python scripts/eval_loss.py +experiment/eval_loss=latent_loss loss_ckpt_path=checkpoints/latent_siamese_edge_grayscale_multi_view_256.ckpt +data/variants=eval_grayscale +data/variants=eval_sketch


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

optimize_sketch_couch:
	python scripts/optimize_sketch.py +experiment/optimize_sketch=couch_latent

