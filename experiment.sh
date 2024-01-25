#!/bin/bash

# settings for old decoder
# python scripts/optimize_normals.py \
#  ckpt_path=/shared/logs/deepsdf.ckpt \
#  +data=shapenet_chair_1024 obj_ids=["1e6f06d182094d4ffebad4f49b26ec52"] \
#  trainer.max_epochs=7 \
#  logger=wandb \
#  +model.video_capture_rate=4 \
#  +model.video_azim=30 \
#  +model.video_elev=15

# settings for new decoder
python scripts/optimize_normals.py \
 ckpt_path=/shared/logs/deepsdf.ckpt \
 +data=shapenet_chair_1024 obj_ids=["1e6f06d182094d4ffebad4f49b26ec52"] \
 trainer.max_epochs=1 \
 model.optimizer.lr=5e-3 \
 trainer.accumulate_grad_batches=4 \
 model.reg_weight=1e-1 \
#  +model.video_capture_rate=8 \
#  +model.video_azim=30 \
#  +model.video_elev=15 \
#  logger=wandb \