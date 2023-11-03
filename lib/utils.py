import logging
from logging import Logger

import pandas as pd
from hydra import compose, initialize
from omegaconf import DictConfig


def create_logger(name: str) -> Logger:
    """
    Create and configure a custom logger with the given name.

    Parameters:
        name (str): The name of the logger. It helps identify the logger when used in
            different parts of the application.

    Returns:
        logging.Logger: A configured logger object that can be used to log messages.

    Usage:
        Use this function to create custom loggers with different names and settings
        throughout your application. Each logger can be accessed using its unique name.

    Example:
        >>> my_logger = create_logger("my_logger")
        >>> my_logger.debug("This is a debug message")
        >>> my_logger.info("This is an info message")
        >>> my_logger.warning("This is a warning message")
        >>> my_logger.error("This is an error message")
        >>> my_logger.critical("This is a critical message")
    """
    # Create a logger with the given name
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Create a log message formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
    )

    # Create a console handler and set the formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    # Return the configured logger
    return logger


def load_config() -> DictConfig:
    """Loads the hydra config via code.

    Returns:
        DictConfig: The initialized config.
    """
    with initialize(config_path="../conf", version_base=None):
        return compose(config_name="config")
