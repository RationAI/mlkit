from collections.abc import Callable
from pathlib import Path
from typing import Any, ParamSpec, cast

import lightning as pl
from lightning import Callback
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT, _PREDICT_OUTPUT
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig


ARTIFACTS_PREFIX = "mlflow-artifacts:/"
ARTIFACTS_DOWNLOAD_PATH = "mlflow_artifacts/checkpoints"


P = ParamSpec("P")


def _copy_kwargs(
    _: Callable[P, None],
) -> Callable[[Callable[..., None]], Callable[P, None]]:
    def return_func(func: Callable[..., None]) -> Callable[P, None]:
        return cast("Callable[P, None]", func)

    return return_func


class Trainer(pl.Trainer):
    @_copy_kwargs(pl.Trainer.__init__)
    def __init__(
        self,
        *,
        callbacks: dict[str, Callback]
        | DictConfig
        | list[Callback]
        | Callback
        | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(callbacks, dict | DictConfig):
            callbacks = list(callbacks.values())
        super().__init__(callbacks=callbacks, **kwargs)

    def _run(
        self, model: pl.LightningModule, ckpt_path: _PATH | None = None
    ) -> _EVALUATE_OUTPUT | _PREDICT_OUTPUT | None:
        if isinstance(ckpt_path, str) and ckpt_path.startswith(ARTIFACTS_PREFIX):
            if not isinstance(self.logger, MLFlowLogger):
                raise ValueError("Cannot download artifacts without MLFlowLogger")

            ckpt_path = download_artifacts(
                ckpt_path,
                dst_path=ARTIFACTS_DOWNLOAD_PATH,
                tracking_uri=self.logger._tracking_uri,
            )

        return super()._run(model, ckpt_path)

    def predict(
        self,
        model: pl.LightningModule | None = None,
        dataloaders: Any | pl.LightningDataModule | None = None,
        datamodule: pl.LightningDataModule | None = None,
        return_predictions: bool
        | None = False,  # override default due to accumulation of results
        ckpt_path: str | Path | None = None,
    ) -> _PREDICT_OUTPUT | None:
        return super().predict(
            model, dataloaders, datamodule, return_predictions, ckpt_path
        )
