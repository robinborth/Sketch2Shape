from typing import List

from hydra import compose, initialize
from hydra.utils import instantiate
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from lib.logger import create_logger

log = create_logger("utils")


def load_config() -> DictConfig:
    """Loads the hydra config via code.

    Returns:
        DictConfig: The initialized config.
    """
    with initialize(config_path="../conf", version_base=None):
        return compose(config_name="config")


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if callbacks_cfg is None:
        log.info("No callbacks specified.")
        return callbacks

    for callback in callbacks_cfg.values():
        if "_target_" in callback.keys():
            callbacks.append(instantiate(callback))

    return callbacks


def instantiate_loggers(loggers_cfg: DictConfig) -> List[Logger]:
    loggers: List[Logger] = []

    if loggers_cfg is None:
        log.info("No loggers specified.")
        return loggers

    for logger in loggers_cfg.values():
        if "_target_" in logger.keys():
            loggers.append(instantiate(logger))
    return loggers
