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
	# python scripts/eval_siamese.py +experiment/eval_siamese=siamese

train_deepsdf:
	python scripts/train_deepsdf.py +experiment/train_deepsdf=shapenet_chair_4096_curriculum

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
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=train_mesh_ckpt_3000
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_val_ckpt_500
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_val_ckpt_1000
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_val_ckpt_1500
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_val_ckpt_2000
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_val_ckpt_2500
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_val_ckpt_3000

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

siamese_ckpt: 
	python scripts/eval_siamese.py +experiment/eval_siamese=siamese ckpt_path=/home/borth/sketch2shape/logs/train_siamese/runs/2024-02-11_17-41-22/checkpoints/epoch_099.ckpt  tags=["eval_siamese","shapenet_chair_4096","siamese_099"]
	python scripts/eval_siamese.py +experiment/eval_siamese=siamese ckpt_path=/home/borth/sketch2shape/logs/train_siamese/runs/2024-02-11_17-41-22/checkpoints/epoch_199.ckpt  tags=["eval_siamese","shapenet_chair_4096","siamese_199"]
	python scripts/eval_siamese.py +experiment/eval_siamese=siamese ckpt_path=/home/borth/sketch2shape/logs/train_siamese/runs/2024-02-11_17-41-22/checkpoints/epoch_299.ckpt  tags=["eval_siamese","shapenet_chair_4096","siamese_299"]
	python scripts/eval_siamese.py +experiment/eval_siamese=siamese ckpt_path=/home/borth/sketch2shape/logs/train_siamese/runs/2024-02-11_17-41-22/checkpoints/epoch_399.ckpt  tags=["eval_siamese","shapenet_chair_4096","siamese_399"]
	python scripts/eval_siamese.py +experiment/eval_siamese=siamese ckpt_path=/home/borth/sketch2shape/logs/train_siamese/runs/2024-02-11_17-41-22/checkpoints/epoch_499.ckpt  tags=["eval_siamese","shapenet_chair_4096","siamese_499"]
	# python scripts/eval_siamese.py +experiment/eval_siamese=siamese ckpt_path=/home/borth/sketch2shape/logs/train_siamese/runs/2024-02-11_17-41-22/checkpoints/epoch_599.ckpt  tags=["eval_siamese","shapenet_chair_4096","siamese_599"]
	# python scripts/eval_siamese.py +experiment/eval_siamese=siamese ckpt_path=/home/borth/sketch2shape/logs/train_siamese/runs/2024-02-11_17-41-22/checkpoints/epoch_699.ckpt  tags=["eval_siamese","shapenet_chair_4096","siamese_699"]
	# python scripts/eval_siamese.py +experiment/eval_siamese=siamese ckpt_path=/home/borth/sketch2shape/logs/train_siamese/runs/2024-02-11_17-41-22/checkpoints/epoch_799.ckpt  tags=["eval_siamese","shapenet_chair_4096","siamese_799"]
	# python scripts/eval_siamese.py +experiment/eval_siamese=siamese ckpt_path=/home/borth/sketch2shape/logs/train_siamese/runs/2024-02-11_17-41-22/checkpoints/epoch_899.ckpt  tags=["eval_siamese","shapenet_chair_4096","siamese_899"]
	# python scripts/eval_siamese.py +experiment/eval_siamese=siamese ckpt_path=/home/borth/sketch2shape/logs/train_siamese/runs/2024-02-11_17-41-22/checkpoints/epoch_999.ckpt  tags=["eval_siamese","shapenet_chair_4096","siamese_999"]
	# python scripts/eval_siamese.py +experiment/eval_siamese=siamese ckpt_path=/home/borth/sketch2shape/logs/train_siamese/runs/2024-02-11_17-41-22/checkpoints/epoch_1099.ckpt  tags=["eval_siamese","shapenet_chair_4096","siamese_1099"]
	# python scripts/eval_siamese.py +experiment/eval_siamese=siamese ckpt_path=/home/borth/sketch2shape/logs/train_siamese/runs/2024-02-11_17-41-22/checkpoints/epoch_1199.ckpt  tags=["eval_siamese","shapenet_chair_4096","siamese_1199"]
	# python scripts/eval_siamese.py +experiment/eval_siamese=siamese ckpt_path=/home/borth/sketch2shape/logs/train_siamese/runs/2024-02-11_17-41-22/checkpoints/epoch_1299.ckpt  tags=["eval_siamese","shapenet_chair_4096","siamese_1299"]
	# python scripts/eval_siamese.py +experiment/eval_siamese=siamese ckpt_path=/home/borth/sketch2shape/logs/train_siamese/runs/2024-02-11_17-41-22/checkpoints/epoch_1399.ckpt  tags=["eval_siamese","shapenet_chair_4096","siamese_1399"]
	# python scripts/eval_siamese.py +experiment/eval_siamese=siamese ckpt_path=/home/borth/sketch2shape/logs/train_siamese/runs/2024-02-11_17-41-22/checkpoints/epoch_1499.ckpt  tags=["eval_siamese","shapenet_chair_4096","siamese_1499"]
	# python scripts/eval_siamese.py +experiment/eval_siamese=siamese ckpt_path=/home/borth/sketch2shape/logs/train_siamese/runs/2024-02-11_17-41-22/checkpoints/epoch_1599.ckpt  tags=["eval_siamese","shapenet_chair_4096","siamese_1599"]
	# python scripts/eval_siamese.py +experiment/eval_siamese=siamese ckpt_path=/home/borth/sketch2shape/logs/train_siamese/runs/2024-02-11_17-41-22/checkpoints/epoch_1699.ckpt  tags=["eval_siamese","shapenet_chair_4096","siamese_1699"]
	# python scripts/eval_siamese.py +experiment/eval_siamese=siamese ckpt_path=/home/borth/sketch2shape/logs/train_siamese/runs/2024-02-11_17-41-22/checkpoints/epoch_1799.ckpt  tags=["eval_siamese","shapenet_chair_4096","siamese_1799"]
	# python scripts/eval_siamese.py +experiment/eval_siamese=siamese ckpt_path=/home/borth/sketch2shape/logs/train_siamese/runs/2024-02-11_17-41-22/checkpoints/epoch_1899.ckpt  tags=["eval_siamese","shapenet_chair_4096","siamese_1899"]
	# python scripts/eval_siamese.py +experiment/eval_siamese=siamese ckpt_path=/home/borth/sketch2shape/logs/train_siamese/runs/2024-02-11_17-41-22/checkpoints/epoch_1999.ckpt  tags=["eval_siamese","shapenet_chair_4096","siamese_1999"]