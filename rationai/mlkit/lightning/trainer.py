import lightning as pl
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT, _PREDICT_OUTPUT
from mlflow.artifacts import download_artifacts


ARTIFACTS_PREFIX = "mlflow-artifacts:/"
ARTIFACTS_DOWNLOAD_PATH = "mlflow_artifacts/checkpoints"


class Trainer(pl.Trainer):
    def __init__(self, *args, **kwargs):
        """A wrapper around the lightning.Trainer class that allows for callbacks to be passed as a dict.

        This allows simpler callbacks configuration using Hydra
        """
        if "callbacks" in kwargs and isinstance(kwargs["callbacks"], dict):
            kwargs["callbacks"] = list(kwargs["callbacks"].values())
        super().__init__(*args, **kwargs)

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
