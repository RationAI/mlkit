import logging
import os
import tempfile
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

import git
import mlflow
from hydra.core.hydra_config import HydraConfig
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.pytorch import loggers
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from mlflow import MlflowClient
from mlflow.utils.mlflow_tags import (
    MLFLOW_GIT_BRANCH,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_REPO_URL,
    MLFLOW_PARENT_RUN_ID,
    MLFLOW_SOURCE_NAME,
)
from omegaconf import DictConfig, OmegaConf


MLFLOW_CHECKPOINT_PATH = "checkpoints"
MLFLOW_CONSOLE_LOG = "console.log"


log = logging.getLogger(__name__)


class MLFlowLogger(loggers.MLFlowLogger):
    def __init__(
        self,
        tags: dict[str, Any] | None = None,
        log_model: Literal[True, False, "all"] = "all",
        log_system_metrics: bool = True,
        **kwargs: Any,
    ) -> None:
        tags = dict(tags or {})  # required because of omegaconf
        tags[MLFLOW_SOURCE_NAME] = os.getenv("HYDRA_SOURCE_TAG", "unstaged")

        super().__init__(
            tags=tags | get_git_tags() | ContextVar(MLFLOW_PARENT_RUN_ID).get({}),
            log_model=log_model,
            **kwargs,
        )
        self.log_system_metrics = log_system_metrics

    @property
    @rank_zero_experiment
    def experiment(self) -> MlflowClient:
        if not self._initialized:
            exp = super().experiment
            mlflow.start_run(self.run_id, log_system_metrics=self.log_system_metrics)
            return exp

        return super().experiment

    @rank_zero_experiment
    def get_stream_logger(self) -> Callable[[str], None]:
        return (
            lambda text: self.experiment.log_text(
                self._run_id, text, MLFLOW_CONSOLE_LOG
            )
            if self._initialized
            else None
        )

    def log_config(self, config: DictConfig) -> None:
        """Logs the configuration to MLFlow."""
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            with open(tmp_dir / "hydra.yaml", "w", encoding="utf-8") as file:
                OmegaConf.save(HydraConfig.get(), file)

            with open(tmp_dir / "config.yaml", "w", encoding="utf-8") as file:
                OmegaConf.save(config, file)

            with open(tmp_dir / "config-resolved.yaml", "w", encoding="utf-8") as file:
                OmegaConf.save(config, file, resolve=True)

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


def get_git_tags() -> dict[str, Any]:
    repo = git.Repo(path=os.getenv("ORIG_WORKING_DIR", os.getcwd()))
    if repo.head.is_detached:
        log.warning("Cannot get git branch ('detached HEAD' state)")

    return {
        MLFLOW_GIT_COMMIT: repo.head.commit.hexsha,
        MLFLOW_GIT_REPO_URL: repo.remotes.origin.url,  # not in the UI
        "git.repo_url": repo.remotes.origin.url,
        MLFLOW_GIT_BRANCH: repo.active_branch.name,  # not in the UI
        "git.branch": repo.active_branch.name,
    }
