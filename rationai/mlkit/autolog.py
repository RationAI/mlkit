import logging
import os
import tempfile
from collections.abc import Callable
from functools import partial, wraps
from pathlib import Path
from typing import overload

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.mlkit.stream import StreamCapture


log = logging.getLogger(__name__)


WrapperT = Callable[[DictConfig], None]
FunctionT = Callable[[DictConfig, MLFlowLogger], None]


@overload
def autolog(
    func: FunctionT,
    *,
    log_config: bool = True,
    log_stream: bool = True,
    log_hyperparams: bool = True,
) -> WrapperT: ...


@overload
def autolog(
    func: None = None,
    *,
    log_config: bool = True,
    log_stream: bool = True,
    log_hyperparams: bool = True,
) -> Callable[[FunctionT], WrapperT]: ...


def autolog(
    func: FunctionT | None = None,
    *,
    log_config: bool = True,
    log_stream: bool = True,
    log_hyperparams: bool = True,
) -> WrapperT | Callable[[FunctionT], WrapperT]:
    """Decorator for automatic logging.

    Args:
        func: The function to decorate
        log_config: Whether to log the hydra configuration files
        log_stream: Whether to log the std streams
        log_hyperparams: Whether to log the hyperparameters defined in the
            config.metadata.hyperparams.
    """
    if func is None:
        return partial(autolog, log_config=log_config, log_stream=log_stream)

    @wraps(func)
    def wrapper(config: DictConfig) -> None:
        logger: MLFlowLogger = hydra.utils.instantiate(config.logger)

        if log_config:
            _log_config(config, logger)

        if (
            log_hyperparams
            and hasattr(config, "metadata")
            and hasattr(config.metadata, "hyperparams")
        ):
            logger.log_hyperparams(config.metadata.hyperparams)

        if log_stream:
            with StreamCapture(logger):
                return func(config, logger)

        return func(config, logger)

    return wrapper


def _log_config(config: DictConfig, logger: MLFlowLogger) -> None:
    """Logs the hydra config."""
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        with open(tmp_dir / "hydra.yaml", "w", encoding="utf-8") as file:
            OmegaConf.save(HydraConfig.get(), file)

        with open(tmp_dir / "config.yaml", "w", encoding="utf-8") as file:
            OmegaConf.save(config, file)

        with open(tmp_dir / "config-resolved.yaml", "w", encoding="utf-8") as file:
            OmegaConf.save(config, file, resolve=True)

        logger.log_artifacts(tmp_dir_str, "configs")
