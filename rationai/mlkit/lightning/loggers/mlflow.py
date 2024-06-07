import os
import tempfile
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from hydra.core.hydra_config import HydraConfig
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.pytorch import loggers
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_SOURCE_NAME
from omegaconf import OmegaConf

from rationai.mlkit.lightning.loggers.get_git_tags import get_git_tags


if TYPE_CHECKING:
    from mlflow import MlflowClient


MLFLOW_CHECKPOINT_PATH = "checkpoints"
MLFLOW_CONSOLE_LOG = "console.log"


class MLFlowLogger(loggers.MLFlowLogger):
    def __init__(
        self,
        tags: dict[str, Any] | None = None,
        log_model: Literal[True, False, "all"] = "all",
        **kwargs: Any,
    ) -> None:
        tags = dict(tags or {})  # required because of omegaconf
        tags[MLFLOW_SOURCE_NAME] = os.getenv("HYDRA_SOURCE_TAG", "unstaged")

        super().__init__(
            tags=tags | get_git_tags() | ContextVar(MLFLOW_PARENT_RUN_ID).get({}),
            log_model=log_model,
            **kwargs,
        )

    @property
    @rank_zero_experiment
    def experiment(self) -> "MlflowClient":
        import ray
        from ray import train
        from ray.air.integrations.mlflow import setup_mlflow

        if not ray.is_initialized():
            return super().experiment

        if self._initialized:
            return self._mlflow_client

        mlflow = setup_mlflow(
            tracking_uri=self._tracking_uri,
            experiment_name=self._experiment_name,
            artifact_location=self._artifact_location,
            run_name=train.get_context().get_experiment_name(),
            create_experiment_if_not_exists=True,
            tags=self.tags,
        )

        _experiment = mlflow.get_experiment_by_name(self._experiment_name)
        _run = mlflow.active_run()

        self._experiment_id = _experiment.experiment_id if _experiment else None
        self._run_id = _run.run_id if _run else None

        self._initialized = True
        return self._mlflow_client

    @rank_zero_experiment
    def get_stream_logger(self) -> Callable[[str], None]:
        return (
            lambda text: self.experiment.log_text(
                self._run_id, text, MLFLOW_CONSOLE_LOG
            )
            if self._initialized
            else None
        )

    def log_config(self, config: dict[str, Any]) -> None:
        """Logs the configuration to MLFlow."""
        with tempfile.TemporaryDirectory(
            prefix="test", suffix="test", dir=os.getcwd()
        ) as tmp_dir:
            with open(f"{tmp_dir}/hydra.yaml", "w") as tmp_file_config:
                OmegaConf.save(HydraConfig.get(), tmp_file_config)

            with open(f"{tmp_dir}/config.yaml", "w") as tmp_file_config:
                OmegaConf.save(config, tmp_file_config)

            with open(f"{tmp_dir}/config-resolved.yaml", "w") as tmp_file_config:
                OmegaConf.save(config, tmp_file_config, resolve=True)

            self.experiment.log_artifacts(self._run_id, tmp_dir, "configs")

    def _scan_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> dict[str, str]:
        checkpoints: dict[str, str] = {}
        if checkpoint_callback.last_model_path:
            checkpoints[Path(checkpoint_callback.last_model_path).stem] = (
                checkpoint_callback.last_model_path
            )

        if checkpoint_callback.best_model_path:
            checkpoints[Path(checkpoint_callback.best_model_path).stem] = (
                checkpoint_callback.best_model_path
            )

        for path in checkpoint_callback.best_k_models:
            checkpoints[Path(path).stem] = path

        return checkpoints

    # Ensures that MLFlow logged checkpoints are in sync with those saved by the trainer.
    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        """Scan checkpoints and log them to MLFlow if not already logged.

        Unfortunately MLFlow supports only deletion of directories, not individual files,
        thereofre the infdividual checkpoints are logged in separate directories.
        """
        checkpoints = self._scan_checkpoints(checkpoint_callback)

        logged_checkpoints = {
            Path(x.path).stem: x.path
            for x in self.experiment.list_artifacts(
                self._run_id, path=MLFLOW_CHECKPOINT_PATH
            )
        }

        # Delete old MLFlow checkpoints (those no logner kept by trainer)
        for key in set(logged_checkpoints).difference(checkpoints):
            self.experiment._tracking_client._get_artifact_repo(
                self._run_id
            ).delete_artifacts(logged_checkpoints[key])

        # Log new checkpoints to MLFlow
        for key in set(checkpoints).difference(logged_checkpoints):
            # Log the checkpoint
            self.experiment.log_artifact(
                self._run_id, checkpoints[key], f"{MLFLOW_CHECKPOINT_PATH}/{key}"
            )

    def log_table(self, data: dict[str, Any], artifact_file: str) -> None:
        """Logs a json table to mlflow as an artifact that can be viewed in the mlflow evaluation.

        Individual logs are appended to the same file.

        Example:
        ```python
        table = {
            "slide_id": 1,
            "acc": 0.5,
        }
        table2 = {
            "slide_id": 2,
            "acc": 0.8,
        }

        self.log_table(data=table, artifact_file="results.json")
        self.log_table(data=table2, artifact_file="results.json")
        ```
        """
        self.experiment.log_table(
            data=data, artifact_file=artifact_file, run_id=self.run_id
        )
