import logging
import os
import tempfile
from collections.abc import Callable
from functools import partial, wraps
from pathlib import Path
from typing import overload

import hydra
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import Logger, MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from rationai.mlkit.stream import StreamCapture, StreamLogger


log = logging.getLogger(__name__)


WrapperT = Callable[[DictConfig], None]
FunctionT = Callable[[DictConfig, Logger], None]


@overload
def autolog(
    func: FunctionT,
    *,
    log_config: bool = True,
    log_stream: bool = True,
) -> WrapperT: ...


@overload
def autolog(
    func: None = None, *, log_config: bool = True, log_stream: bool = True
) -> Callable[[FunctionT], WrapperT]: ...


def autolog(
    func: FunctionT | None = None,
    *,
    log_config: bool = True,
    log_stream: bool = True,
) -> WrapperT | Callable[[FunctionT], WrapperT]:
    """Decorator for automatic logging.

    Args:
        func: The function to decorate
        log_config: Whether to log the hydra configuration files
        log_stream: Whether to log the std streams
    """
    if func is None:
        return partial(autolog, log_config=log_config, log_stream=log_stream)

    @wraps(func)
    def wrapper(config: DictConfig) -> None:
        logger = hydra.utils.instantiate(config.logger)

        if log_config:
            _log_config(config, logger)

        if log_stream:
            return _log_stream(logger, partial(func, config, logger))

        return func(config, logger)

    return wrapper


def _log_config(config: DictConfig, logger: Logger) -> None:
    """Logs the hydra config."""
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        with open(tmp_dir / "hydra.yaml", "w", encoding="utf-8") as file:
            OmegaConf.save(HydraConfig.get(), file)

        with open(tmp_dir / "config.yaml", "w", encoding="utf-8") as file:
            OmegaConf.save(config, file)

        with open(tmp_dir / "config-resolved.yaml", "w", encoding="utf-8") as file:
            OmegaConf.save(config, file, resolve=True)

        if isinstance(logger, MLFlowLogger):
            logger.experiment.log_artifacts(logger.run_id, tmp_dir, "configs")
        else:
            log.warning(
                "The %s logger is not supported for logging the configuration",
                logger,
            )


def _log_stream(logger: Logger, func: Callable[[], None]) -> None:
    """Logs the std streams using the logger."""
    if isinstance(logger, StreamLogger):
        with StreamCapture(logger):
            return func()

    log.warning("The %s logger is not supported for logging the std streams", logger)
    return func()
