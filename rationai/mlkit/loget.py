import logging
import os
import tempfile
from collections.abc import Callable
from functools import wraps
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import Logger, MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from rationai.mlkit.stream import StreamCapture, StreamLogger


log = logging.getLogger(__name__)


def loget(func: Callable[[DictConfig, Logger], None]) -> Callable[[DictConfig], None]:
    """Decorator for logging the hydra configuration files and std streams using the logger specified in the configuration."""

    @wraps(func)
    def wrapper(config: DictConfig) -> None:
        logger = hydra.utils.instantiate(config.logger)

        # Save the configuration
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

        # Capture the output
        if isinstance(logger, StreamLogger):
            with StreamCapture(logger):
                return func(config, logger)  # type: ignore[return-value]

        log.warning(
            "The %s logger is not supported for logging the std streams", logger
        )
        return func(config, logger)

    return wrapper
