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

train_deepsdf:
	python scripts/train_deepsdf.py +experiment=deepsdf_overfit_1 logger=wandb
	python scripts/train_deepsdf.py +experiment=deepsdf_overfit_4 logger=wandb
	python scripts/train_deepsdf.py +experiment=deepsdf_overfit_16 logger=wandb

train_siamese:
	# python scripts/train_siamese.py +experiment=siamese
	# python scripts/eval_siamese.py +experiment=eval_resnet
	# python scripts/eval_siamese.py +experiment=eval_clip
	# python scripts/train_siamese.py +experiment=siamese_scale_loss 
	# python scripts/train_siamese.py +experiment=siamese_sample_m data.sampler.m=4
	# python scripts/train_siamese.py +experiment=siamese_sample_m data.sampler.m=8
	python scripts/train_siamese.py +experiment=siamese_mine_full_batch
	python scripts/train_siamese.py +experiment=siamese_pretrained
	python scripts/train_siamese.py +experiment=siamese_type_of_triplets

eval_siamese:
	python scripts/eval_siamese.py +experiment=eval_resnet18
	python scripts/eval_siamese.py +experiment=eval_clip

