from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import mlflow
import mlflow.entities
from mlflow import MlflowClient, MlflowException
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from ray.train import RunConfig
from ray.train.base_trainer import BaseTrainer
from ray.tune import Trainable, TuneConfig
from ray.tune import Tuner as RayTuner

from rationai.mlkit.lightning.loggers.get_git_tags import get_git_tags


@dataclass
class MLFlowConfig:
    tracking_uri: str
    experiment_name: str
    run_name: str
    tags: dict[str, Any]


class Tuner(RayTuner):
    def __init__(
        self,
        trainable: str | Callable | type[Trainable] | BaseTrainer | None = None,
        *,
        param_space: dict[str, Any] | None = None,
        mlflow_config: MLFlowConfig | None = None,
        tune_config: TuneConfig | None = None,
        run_config: RunConfig | None = None,
    ) -> None:
        super().__init__(
            trainable=trainable,
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config,
        )

        self.mlflow_config = mlflow_config
        if self._parent_run is not None:
            ContextVar(MLFLOW_PARENT_RUN_ID).set(self._setup_parent_run.info.run_id)

    @cached_property
    def _parent_run(self) -> mlflow.entities.Run | None:
        if not self.mlflow_config:
            return None

        mlflow.set_tracking_uri(self.ml.tracking_uri)
        client = MlflowClient()
        try:
            experiment = client.create_experiment(self.mlflow_config.experiment_name)
        except MlflowException:
            experiment = client.get_experiment_by_name(
                self.mlflow_config.experiment_name
            ).experiment_id

        return client.create_run(
            experiment_id=experiment,
            run_name=self.mlflow_config.run_name,
            tags=self.mlflow_config.tags | get_git_tags(),
        )
