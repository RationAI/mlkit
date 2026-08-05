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
from typing import Any

import mlflow
from lightning.pytorch.callbacks import Callback

from rationai.mlkit.provenance.environment import (
    capture_environment,
)


log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Callback
# ──────────────────────────────────────────────


class EnvironmentCallback(Callback):
    """Capture hardware, docker, user, and environment snapshot at training start.

    Stores results on ``self`` so sibling callbacks (e.g. ``ProvenanceCallback``)
    can read them without duplicating work.

    Attributes set after ``on_fit_start``:
        - ``_hardware`` (dict)
        - ``_pytorch`` (dict)
        - ``_docker`` (dict)
        - ``_seeds`` (dict)
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
        self._hardware: dict[str, str | int] = {}
        self._pytorch: dict[str, str] = {}
        self._docker: dict[str, str | bool] = {}
        self._seeds: dict[str, str] = {}
        self._frozen_requirements: str | None = None
        self._user_run_id: str | None = None
        self._user_tags: dict[str, str] = {}
        self._temp_dirs: list[str] = []

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Capture environment metadata at the start of training."""
        if not mlflow.active_run():
            return

        # ── Hardware skip check (needs trainer.loggers) ─────────
        sys_metrics_on = any(
            getattr(logger, "log_system_metrics", False) for logger in trainer.loggers
        )
        if sys_metrics_on and not self.skip_hardware:
            log.info(
                "[EnvironmentCallback] Skipping hardware — MLflow system metrics enabled"
            )

        # ── Capture environment (delegates to provenance module) ──
        result = capture_environment(
            skip_hardware=self.skip_hardware or sys_metrics_on,
            snapshot_env=self.snapshot_env,
            strict=self.strict,
        )

        # ── Populate instance fields for sibling callbacks ──────
        self._user_run_id = result.get("user_run_id")  # type: ignore[assignment]
        self._user_tags = result.get("user_tags") or {}  # type: ignore[assignment]
        self._hardware = result.get("hardware") or {}  # type: ignore[assignment]
        self._pytorch = result.get("pytorch") or {}  # type: ignore[assignment]
        self._docker = result.get("docker") or {}  # type: ignore[assignment]
        self._seeds = result.get("seeds") or {}  # type: ignore[assignment]
        self._frozen_requirements = result.get("frozen_requirements")  # type: ignore[assignment]
