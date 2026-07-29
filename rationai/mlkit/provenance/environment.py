"""Environment detection helpers for provenance tracking.

Provides hardware, docker, user lookup, and environment snapshot functions
used by both callbacks and standalone provenance workflows.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import platform
import shutil
import subprocess

import torch


# ──────────────────────────────────────────────
# User lookup
# ──────────────────────────────────────────────


def _lookup_user_run() -> tuple[str | None, dict[str, str]]:
    """Find the user run from User_Registry.  Auto-detect username."""
    import mlflow
    import pandas as pd

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


# ──────────────────────────────────────────────
# Hardware detection
# ──────────────────────────────────────────────


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


# ──────────────────────────────────────────────
# Docker detection
# ──────────────────────────────────────────────


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


# ──────────────────────────────────────────────
# Environment snapshot
# ──────────────────────────────────────────────


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
