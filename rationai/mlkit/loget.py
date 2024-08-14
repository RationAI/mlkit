import logging
import os
import sys
import tempfile
from functools import wraps
from pathlib import Path
from typing import TextIO

import hydra
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from rationai.mlkit.stream import StreamCapture, StreamLogger


log = logging.getLogger(__name__)


def loget(streams: tuple[TextIO, ...] = (sys.stdout, sys.stderr)):
    """Logs the omegaconf configuration files.

    Note:
        Currently only MLFlowLogger is supported for logging the configuration.
    """

    def wrapped(func):
        @wraps(func)
        def wrapper(config: DictConfig) -> DictConfig:
            logger = hydra.utils.instantiate(config.logger)
            config.logger = logger

            # Save the configuration
            with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_dir_str:
                tmp_dir = Path(tmp_dir_str)
                with open(tmp_dir / "hydra.yaml", "w", encoding="utf-8") as file:
                    OmegaConf.save(HydraConfig.get(), file)

                with open(tmp_dir / "config.yaml", "w", encoding="utf-8") as file:
                    OmegaConf.save(config, file)

                with open(
                    tmp_dir / "config-resolved.yaml", "w", encoding="utf-8"
                ) as file:
                    OmegaConf.save(config, file, resolve=True)

                if isinstance(logger, MLFlowLogger):
                    logger.experiment.log_artifacts(logger.run_id, tmp_dir, "configs")
                else:
                    log.warning(
                        "The logger %s is not supported for logging the configuration",
                        logger,
                    )

            # Capture the output
            if isinstance(logger, StreamLogger):
                with StreamCapture(logger, streams=streams):
                    return func(config)
            return func(config)

        return wrapper

    return wrapped
