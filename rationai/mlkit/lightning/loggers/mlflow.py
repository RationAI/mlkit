import os
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Literal

from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.pytorch import loggers
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_SOURCE_NAME

from rationai.mlkit.lightning.loggers.get_git_tags import get_git_tags


if TYPE_CHECKING:
    from mlflow import MlflowClient


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

        self._experiment_id = mlflow.get_experiment_by_name(
            self._experiment_name
        ).experiment_id
        self._run_id = mlflow.active_run().info.run_id

        self._initialized = True
        return self._mlflow_client
