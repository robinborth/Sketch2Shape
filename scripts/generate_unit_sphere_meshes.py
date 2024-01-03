# import glob
# import json
# import os
# from pathlib import Path

# import hydra
# import numpy as np
# import trimesh
# from omegaconf import DictConfig, OmegaConf


# @hydra.main(version_base=None, config_path="../conf", config_name="train_deepsdf")
# def get_files(cfg: DictConfig) -> None:
#     # cfg.data.data_dir = "/home/korth/sketch2shape/data/deepsdf/SdfSamples/overfit_batch"
#     # cfg.data.norm_dir = (
#     #     "/home/korth/sketch2shape/data/deepsdf/NormalizationParameters/overfit_batch"
#     # )
#     if not os.path.exists(cfg.data.data_dir):
#         raise ValueError(
#             "Please provide data directory, e.g. using ++data.data_dir=/path/to/dir"
#         )

#     file_ids = list()
#     for file in glob.glob(cfg.data.data_dir + "/*.npz"):
#         file_ids.append(file.split("/")[-1][:-4])

#     for file_id in file_ids:
#         norm = np.load(cfg.data.norm_dir + "/" + file_id + ".npz", allow_pickle=True)
#         obj_path = glob.glob(cfg.paths.shapenet_dir + "/**model_normalized.obj")


# if __name__ == "__main__":
#     get_files()
