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

main_demo:
	python scripts/create_video.py +experiment/create_video=main_demo

multi_view:
	python scripts/optimize_sketch.py +experiment/create_video=multi_view_demo

########################################################################
# Debug
########################################################################

traverse_latent:
	python scripts/traverse_latent.py +experiment/traverse_latent=train_train_1

optimize_deepsdf:
	python scripts/optimize_deepsdf.py +experiment/optimize_deepsdf=mean_train

optimize_normals:
	python scripts/optimize_normals.py +experiment/optimize_normals=mean_train

########################################################################
# Train 
########################################################################

train_deepsdf:
	python scripts/train_deepsdf.py +experiment/train_deepsdf=shapenet_chair_4096

train_latent_loss:
	python scripts/train_loss.py +experiment/train_loss=latent_traverse

train_triplet_loss:
	python scripts/train_loss.py +experiment/train_loss=triplet_traverse

########################################################################
# Eval 
########################################################################

eval_retrieval:
	python scripts/optimize_deepsdf.py +experiment/eval_deepsdf=triplet_traverse

eval_encoder:
	python scripts/optimize_deepsdf.py +experiment/eval_deepsdf=latent_traverse

eval_optimize:
	python scripts/optimize_sketch.py +experiment/optimize_sketch=global

eval_loss:
	python scripts/eval_loss.py +experiment/eval_loss=resnet18
	python scripts/eval_loss.py +experiment/eval_loss=clip
	python scripts/eval_loss.py +experiment/eval_loss=triplet_loss

########################################################################
# Report
########################################################################

train_loss_report:
	python scripts/train_loss.py +experiment/train_loss=latent_synthetic
	python scripts/train_loss.py +experiment/train_loss=latent_rendered
	python scripts/train_loss.py +experiment/train_loss=latent_traverse
	python scripts/train_loss.py +experiment/train_loss=triplet_traverse

eval_baseline_report:
	python scripts/optimize_deepsdf.py +experiment/eval_deepsdf=latent_rendered
	python scripts/optimize_deepsdf.py +experiment/eval_deepsdf=latent_synthetic 
	python scripts/optimize_deepsdf.py +experiment/eval_deepsdf=triplet_traverse
	python scripts/optimize_deepsdf.py +experiment/eval_deepsdf=latent_traverse
	python scripts/optimize_deepsdf.py +experiment/eval_deepsdf=triplet_traverse model.prior_mode=9 model.prior_view_id=6
	python scripts/optimize_deepsdf.py +experiment/eval_deepsdf=latent_traverse model.prior_mode=9 model.prior_view_id=6

eval_optimize_report:
	python scripts/optimize_sketch.py +experiment/optimize_sketch=regularize
	python scripts/optimize_sketch.py +experiment/optimize_sketch=regularize mode=10 view_id=0
	python scripts/optimize_sketch.py +experiment/optimize_sketch=global
	python scripts/optimize_sketch.py +experiment/optimize_sketch=global mode=10 view_id=0
