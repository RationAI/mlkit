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

import contextlib
import hashlib
import logging
import os
import platform
import shutil
import subprocess
import uuid
from datetime import UTC, datetime
from typing import Any

import mlflow
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback


log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _lookup_user_run() -> tuple[str | None, dict[str, str]]:
    """Find the user run from User_Registry.  Auto-detect username."""
    from rationai.mlkit.provenance.dataset import _lookup_experiment

    username = os.environ.get("MLFLOW_USER")
    if not username:
        with contextlib.suppress(subprocess.CalledProcessError):
            username = (
                subprocess.check_output(
                    ["git", "config", "user.name"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
    if not username:
        username = os.environ.get("USER", "unknown")

    exp_id = _lookup_experiment("User_Registry")
    if exp_id is None:
        return None, {}

    _runs_df = mlflow.search_runs(experiment_ids=[exp_id])
    runs_df: pd.DataFrame = _runs_df  # search_runs may return RunList in old mlflow
    if runs_df.empty:
        return None, {}

    matched = runs_df[runs_df["tags.username"] == username]
    if matched.empty:
        matched = runs_df.head(1)

    row = matched.iloc[0]
    run_obj = mlflow.get_run(row.run_id)
    return row.run_id, dict(run_obj.data.tags)


def _detect_hardware() -> dict[str, str | int]:
    """Detect CPU/GPU/hardware info."""
    info: dict[str, str | int] = {}

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        cap = torch.cuda.get_device_capability(0)
        info["gpu_compute_capability"] = f"{cap[0]}.{cap[1]}"
        info["cuda_version"] = torch.version.cuda or "unknown"
    else:
        info["gpu_name"] = "none"

    info["cpu_count_logical"] = os.cpu_count() or 0
    info["os_platform"] = platform.platform()
    info["python_version"] = platform.python_version()

    try:
        import psutil

        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / 1e9, 1)
    except ImportError:
        pass

    return info


def _detect_docker() -> dict[str, str | bool]:
    """Detect if running inside Docker and extract container info."""
    info: dict[str, str | bool] = {"docker": False}

    if os.path.exists("/.dockerenv"):
        info["docker"] = True

    if not info["docker"]:
        try:
            with open("/proc/self/cgroup") as f:
                for line in f:
                    for p in line.strip().split("/"):
                        if len(p) >= 12 and all(
                            c in "0123456789abcdef" for c in p[:12]
                        ):
                            info["docker"] = True
                            info["container_id_short"] = p[:12]
                            break
        except FileNotFoundError:
            pass

    if not info.get("container_id_short"):
        try:
            with open("/proc/self/mountinfo") as f:
                for line in f:
                    for p in line.split():
                        if len(p) == 64 and all(c in "0123456789abcdef" for c in p):
                            info["container_id_short"] = p[:12]
                            break
        except FileNotFoundError:
            pass

    if info["docker"]:
        cid = str(info.get("container_id_short", ""))
        if cid:
            try:
                result = subprocess.run(
                    ["docker", "inspect", "--format={{.Config.Image}}", cid],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    image = result.stdout.strip()
                    info["docker_image"] = image
                    info["docker_image_hash"] = hashlib.sha256(
                        image.encode()
                    ).hexdigest()[:16]
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

    return info


def _snapshot_environment(artifact_dir: str) -> str:
    """Freeze environment to *artifact_dir* and return the pip-freeze text."""
    req_path = os.path.join(artifact_dir, "requirements_frozen.txt")
    with open(req_path, "w") as f:
        subprocess.run(["uv", "pip", "freeze"], stdout=f, check=True)

    for src in ("pyproject.toml", "uv.lock"):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(artifact_dir, src))

    with open(req_path) as f:
        return f.read()


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
