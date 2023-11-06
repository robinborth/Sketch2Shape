from pathlib import Path

import numpy as np

from lib.data.sdf_utils import get_sdf_samples
from lib.utils import load_config

if __name__ == "__main__":
    cfg = load_config()
    p = Path(cfg.data_path)

    for file in p.glob("**/*.obj"):
        np_data = get_sdf_samples(file, number_of_points=cfg.data.number_of_points)
        save_path = Path(file.parents[1], f"sdf_points_{cfg.data.number_of_points}")
        np.save(save_path, np_data, allow_pickle=True)
        print(f"extrackted points and stored them at: {save_path}.npy")
