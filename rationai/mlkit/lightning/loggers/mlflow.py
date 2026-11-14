import logging
import os
import shutil
import tempfile
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

import git
import git.exc
import hydra
import mlflow
import torch
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.pytorch import loggers
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers.utilities import _scan_checkpoints
from mlflow import MlflowClient
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.pytorch import FLAVOR_NAME
from mlflow.utils.mlflow_tags import (
    MLFLOW_GIT_BRANCH,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_REPO_URL,
    MLFLOW_PARENT_RUN_ID,
    MLFLOW_SOURCE_NAME,
)
from torch import Tensor

from rationai.mlkit.stream import StreamLogger


MLFLOW_CHECKPOINT_PATH = "checkpoints"
MLFLOW_CONSOLE_LOG = "console.log"
MLFLOW_CHECKPOINT_FILE_NAME = "checkpoint.ckpt"


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

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self.experiment.log_artifact(self.run_id, local_path, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None) -> None:
        self.experiment.log_artifacts(self.run_id, local_dir, artifact_path)

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

    def _log_checkpoint(
        self, key: str, path: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Log a lightning checkpoint to MLFlow and register it as a model.

        Args:
            key: The key to identify the checkpoint in MLFlow.
            path: The path to the local checkpoint file.
            metadata: Optional metadata stored in the MLmodel file.
        """
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)

            # Copy the checkpoint to a temporary directory so we can change its name
            shutil.copy(path, tmpdir / MLFLOW_CHECKPOINT_FILE_NAME)

            # Add model manifest for MLFlow to recognize the checkpoint
            mlflow_model = mlflow.models.Model(
                f"{MLFLOW_CHECKPOINT_PATH}/{key}",
                self.run_id,
                flavors={
                    FLAVOR_NAME: {
                        "model_data": MLFLOW_CHECKPOINT_FILE_NAME,
                        "pytorch_version": str(torch.__version__),
                    }
                },
                model_size_bytes=os.path.getsize(path),
                metadata=metadata,
            )
            mlflow_model.save(tmpdir / MLMODEL_FILE_NAME)

            # Log the checkpoint to MLFlow
            self.experiment.log_artifacts(
                self.run_id, tmpdir, f"{MLFLOW_CHECKPOINT_PATH}/{key}"
            )

    # Ensures that MLFlow logged checkpoints are in sync with those saved by the trainer.
    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        """Log new or modified checkpoints to MLFlow and delete those no longer kept.

        A checkpoint is (re-)logged when its file modification time is newer than
        when it was last logged by this logger (Lightning's approach), so checkpoints
        overwritten under the same name (e.g. ``last.ckpt``) are kept up to date.
        """
        # All checkpoints currently kept by the trainer: (mtime, path, score, tag)
        checkpoints = _scan_checkpoints(checkpoint_callback, {})

        for mtime, path, score, _ in checkpoints:
            if self._logged_model_time.get(path, float("-inf")) >= mtime:
                continue

            self._log_checkpoint(
                Path(path).stem,
                path,
                metadata={
                    "score": score.item() if isinstance(score, Tensor) else score,
                    "original_filename": Path(path).name,
                },
            )
            self._logged_model_time[path] = mtime

        # Delete MLFlow checkpoints that are no longer kept by the trainer
        current = {Path(path).stem for _, path, _, _ in checkpoints}
        for artifact in self.experiment.list_artifacts(
            self.run_id, path=MLFLOW_CHECKPOINT_PATH
        ):
            if Path(artifact.path).stem not in current:
                self.experiment._tracking_client._get_artifact_repo(
                    self.run_id
                ).delete_artifacts(artifact.path)

        # Forget local checkpoints that were removed by the trainer
        for path in set(self._logged_model_time).difference(
            path for _, path, _, _ in checkpoints
        ):
            del self._logged_model_time[path]


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
