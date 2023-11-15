import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig

from lib.data.generate import reconstruct_training_data


@hydra.main(version_base=None, config_path="../conf", config_name="config_deefsdf")
def reconstruct(cfg: DictConfig) -> None:
    model = instantiate(cfg.model)
    reconstruct_training_data(model, cfg.generate.checkpoint_path)


if __name__ == "__main__":
    reconstruct()
