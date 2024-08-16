import logging
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

import git
import git.exc
import hydra
import mlflow
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

from rationai.mlkit.stream import StreamLogger


MLFLOW_CHECKPOINT_PATH = "checkpoints"
MLFLOW_CONSOLE_LOG = "console.log"


log = logging.getLogger(__name__)


class MLFlowLogger(loggers.MLFlowLogger, StreamLogger):
    def __init__(
        self,
        tags: dict[str, Any] | None = None,
        log_model: Literal[True, False, "all"] = "all",
        log_system_metrics: bool = False,
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

    def log_stream(self, text: str) -> None:
        self.experiment.log_text(self.run_id, text, MLFLOW_CONSOLE_LOG)

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
                self.run_id, path=MLFLOW_CHECKPOINT_PATH
            )
        }

        # Log new checkpoints to MLFlow
        for key in set(checkpoints).difference(logged_checkpoints):
            # Log the checkpoint
            self.experiment.log_artifact(
                self.run_id, checkpoints[key], f"{MLFLOW_CHECKPOINT_PATH}/{key}"
            )

        # Delete old MLFlow checkpoints (those no logner kept by trainer)
        for key in set(logged_checkpoints).difference(checkpoints):
            self.experiment._tracking_client._get_artifact_repo(
                self.run_id
            ).delete_artifacts(logged_checkpoints[key])

    def log_table(self, data: dict[str, Any], artifact_file: str) -> None:
        """Logs a json table to mlflow as an artifact that can be viewed in the mlflow evaluation.

        Individual logs are appended to the same file.

        Args:
            data: The data to log.
            artifact_file: The name of the artifact file to log to.

        Examples:
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
    try:
        path = hydra.utils.get_original_cwd()
    except ValueError:
        path = os.getenv("ORIG_WORKING_DIR", os.getcwd())

    try:
        repo = git.Repo(path)
    except (git.exc.InvalidGitRepositoryError, git.exc.NoSuchPathError):
        log.warning("Cannot get git tags (not a git repository)")
        return {}

    tags = {
        MLFLOW_GIT_COMMIT: repo.head.commit.hexsha,
    }

    if repo.remotes:
        try:
            remote_url = repo.remotes.origin.url
        except AttributeError:
            remote_url = repo.remotes[0].url
        tags[MLFLOW_GIT_REPO_URL] = remote_url  # not in the UI
        tags["git.repo_url"] = remote_url
    else:
        log.warning("Cannot get git remote url")

    if not repo.head.is_detached:
        tags[MLFLOW_GIT_BRANCH] = repo.active_branch.name  # not in the UI
        tags["git.branch"] = repo.active_branch.name
    else:
        log.warning("Cannot get git branch ('detached HEAD' state)")

    return tags
