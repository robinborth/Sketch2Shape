from typing import Any, Dict, List

from hydra import compose, initialize
from hydra.utils import instantiate
from lightning import Callback
from lightning.pytorch.loggers import Logger
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from lib.logger import create_logger

log = create_logger("utils")


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if callbacks_cfg is None:
        log.info("No callbacks specified.")
        return callbacks

    for callback in callbacks_cfg.values():
        if "_target_" in callback.keys():
            callbacks.append(instantiate(callback))

    return callbacks


def instantiate_loggers(loggers_cfg: DictConfig):
    if loggers_cfg is None:
        log.info("No loggers specified.")
        return None

    for logger in loggers_cfg.values():
        if "_target_" in logger.keys():
            return instantiate(logger)

    return None


def load_config(config_name: str, overrides: list = None) -> DictConfig:
    """Loads the hydra config via code.

    Args:
        config_name (str): The name of the configuration.
        overrides (list): List of overrides to apply.

    Returns:
        DictConfig: The initialized config.
    """
    with initialize(config_path="../conf", version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def log_hyperparameters(object_dict) -> None:
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)
