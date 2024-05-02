import os

import ray
from ray.runtime_env import RuntimeEnv


def setup_ray() -> None:
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"  # disable auto logging
    runtime_env = RuntimeEnv(
        env_vars={
            "ORIG_WORKING_DIR": os.getcwd(),  # for relative paths
            "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "true",  # enable logging system metrics
        }
    )
    ray.init(runtime_env=runtime_env)
