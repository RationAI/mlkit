"""Lightning callback that captures environment provenance (hardware, docker, env snapshot).

Extracted from ProvenanceCallback so users who only need environment metadata
don't have to pull in the full PROV machinery.

Example::

    from rationai.mlkit.lightning.callbacks import EnvironmentCallback

    trainer = Trainer(
        callbacks=[EnvironmentCallback()],
        logger=MLFlowLogger(...),
    )
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any

import mlflow
from lightning.pytorch.callbacks import Callback

from rationai.mlkit.provenance.environment import (
    _detect_docker,
    _detect_hardware,
    _lookup_user_run,
    _snapshot_environment,
)


log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Callback
# ──────────────────────────────────────────────


class EnvironmentCallback(Callback):
    """Capture hardware, docker, git, user, and environment snapshot at training start.

    Stores results on ``self`` so sibling callbacks (e.g. ``ProvenanceCallback``)
    can read them without duplicating work.

    Attributes set after ``on_fit_start``:
        - ``_git_commit``, ``_git_url``, ``_git_branch``
        - ``_hardware`` (dict)
        - ``_docker`` (dict)
        - ``_frozen_requirements`` (str | None)
        - ``_user_run_id``, ``_user_tags``

    Args:
        skip_hardware: Skip hardware detection if True (MLflow system metrics
            are already enabled). Auto-detected from trainer loggers by default.
        snapshot_env: If True, freeze the environment to an MLflow artifact.
        strict: If True, re-raise errors from optional steps instead of logging.
    """

    def __init__(
        self,
        skip_hardware: bool = False,
        snapshot_env: bool = True,
        strict: bool = False,
    ) -> None:
        """Initialise the environment callback.

        Args:
            skip_hardware: Skip hardware detection if True. Auto-detected from
                trainer loggers by default.
            snapshot_env: If True, freeze the environment to an MLflow artifact.
            strict: If True, re-raise errors from optional steps instead of logging.
        """
        self.skip_hardware = skip_hardware
        self.snapshot_env = snapshot_env
        self.strict = strict

        # Populated during on_fit_start
        self._git_commit: str = "unknown"
        self._git_url: str = "unknown"
        self._git_branch: str = "unknown"
        self._hardware: dict[str, str | int] = {}
        self._docker: dict[str, str | bool] = {}
        self._frozen_requirements: str | None = None
        self._user_run_id: str | None = None
        self._user_tags: dict[str, str] = {}
        self._temp_dirs: list[str] = []

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Capture environment metadata at the start of training."""
        if not mlflow.active_run():
            return

        # ── Git info (read from MLflow tags set by MLFlowLogger) ──
        try:
            run = mlflow.active_run()
            git_tags: dict[str, str] = {}
            if run and run.info and run.info.run_id:
                client = mlflow.tracking.MlflowClient()
                run_data = client.get_run(run.info.run_id)
                git_tags = dict(run_data.data.tags) if run_data.data.tags else {}
            self._git_commit = git_tags.get(
                "mlflow.source.git.commit", git_tags.get("git.commit", "unknown")
            )
            self._git_url = git_tags.get(
                "mlflow.source.git.repoUrl", git_tags.get("git.repo_url", "unknown")
            )
            self._git_branch = git_tags.get(
                "mlflow.source.git.branch", git_tags.get("git.branch", "unknown")
            )
        except Exception as e:
            if self.strict:
                raise
            log.warning("[EnvironmentCallback] Git info failed: %s", e)

        # ── User lookup ─────────────────────────────────────────
        try:
            user_run_id, user_tags = _lookup_user_run()
            self._user_run_id = user_run_id
            self._user_tags = user_tags or {}
        except Exception as e:
            if self.strict:
                raise
            log.warning("[EnvironmentCallback] User lookup failed: %s", e)

        # ── Hardware (skip if MLflow system metrics are on) ─────
        if not self.skip_hardware:
            sys_metrics_on = any(
                getattr(logger, "log_system_metrics", False)
                for logger in trainer.loggers
            )
            if sys_metrics_on:
                log.info(
                    "[EnvironmentCallback] Skipping hardware — MLflow system metrics enabled"
                )
            else:
                try:
                    self._hardware = _detect_hardware()
                except Exception as e:
                    if self.strict:
                        raise
                    log.warning(
                        "[EnvironmentCallback] Hardware detection failed: %s", e
                    )

        # ── Docker detection ────────────────────────────────────
        try:
            self._docker = _detect_docker()
        except Exception as e:
            if self.strict:
                raise
            log.warning("[EnvironmentCallback] Docker detection failed: %s", e)

        # ── Log tags (git + user) ───────────────────────────────
        env_tags: dict[str, str] = {}
        if self._user_run_id:
            env_tags["user_run_id"] = self._user_run_id
            for key in ("username", "real_name", "organization"):
                if key in self._user_tags:
                    env_tags[key] = self._user_tags[key]

        from rationai.mlkit.provenance.dataset import _lookup_dataset_run

        dataset_run_id = _lookup_dataset_run()
        if dataset_run_id:
            env_tags["dataset_run_id"] = dataset_run_id

        env_tags.update(
            {
                "git_commit": self._git_commit,
                "git_url": self._git_url,
                "git_branch": self._git_branch,
                "prov_start_time": datetime.now(UTC).isoformat(),
            }
        )
        mlflow.set_tags(env_tags)

        # ── Log hardware + docker params ────────────────────────
        all_params: dict[str, str | float | int] = {**self._hardware, **self._docker}
        if all_params:
            mlflow.log_params(all_params)

        # ── Environment snapshot ────────────────────────────────
        if self.snapshot_env:
            artifact_dir = f"_mlflow_env_{uuid.uuid4().hex[:8]}"
            os.makedirs(artifact_dir, exist_ok=True)
            self._temp_dirs.append(artifact_dir)
            try:
                self._frozen_requirements = _snapshot_environment(artifact_dir)
                mlflow.log_artifacts(artifact_dir, artifact_path="environment")
            except Exception as e:
                if self.strict:
                    raise
                log.warning("[EnvironmentCallback] Environment snapshot failed: %s", e)
