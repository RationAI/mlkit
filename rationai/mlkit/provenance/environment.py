"""Environment detection helpers for provenance tracking.

Provides hardware and environment snapshot functions
used by both callbacks and standalone provenance workflows.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from collections.abc import Callable

import torch


# ──────────────────────────────────────────────
# User lookup
# ──────────────────────────────────────────────


def lookup_user_run() -> tuple[str | None, dict[str, str]]:
    """Find the user run from User_Registry.  Auto-detect username."""
    import mlflow
    import pandas as pd

    from rationai.mlkit.provenance.dataset import lookup_experiment

    username = os.environ.get("MLFLOW_USER")
    if not username:
        raise RuntimeError(
            "MLFLOW_USER environment variable not set. Cannot lookup user run. Please set MLFLOW_USER to your username (e.g. 'jdoe') before running."
        )

    exp_id = lookup_experiment("User_Registry")
    if exp_id is None:
        raise RuntimeError(
            "User_Registry experiment not found. Cannot lookup user run. Please ensure that the User_Registry experiment exists in MLflow."
        )

    runs_df = cast(
        "pd.DataFrame",
        mlflow.search_runs(experiment_ids=[exp_id], output_format="pandas"),
    )
    if runs_df.empty:
        raise RuntimeError(
            "No runs found for user. Cannot lookup user run. Please ensure that the User_Registry experiment has at exactly one run for your username."
        )

    if "tags.username" not in runs_df.columns:
        raise RuntimeError(
            "No 'tags.username' column in User_Registry. No user runs tagged yet."
        )
    matched = runs_df[runs_df["tags.username"] == username]

    if matched.empty:
        raise RuntimeError(
            f"No run found for username '{username}' in User_Registry. Cannot lookup user run. Please ensure that the User_Registry experiment has exactly one run for your username."
        )

    row = matched.iloc[0]
    run_obj = mlflow.get_run(row.run_id)
    return row.run_id, dict(run_obj.data.tags)


# ──────────────────────────────────────────────
# Hardware detection
# ──────────────────────────────────────────────


def read_ram_limit() -> dict[str, float | str]:
    """Read RAM limit from cgroup files (Kubernetes)."""
    CGROUP_UNLIMITED = 9223372036854771712

    mem_max = read_file_int("/sys/fs/cgroup/memory.max")
    if mem_max is None or mem_max >= CGROUP_UNLIMITED:
        mem_max = read_file_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")

    if mem_max is not None and mem_max < CGROUP_UNLIMITED:
        return {"memory_limit_gib": round(mem_max / (1024**3), 2)}

    # Pokud limit není (běh mimo K8s / unlimited), zachováme klíč se známou hodnotou
    return {"memory_limit_gib": "unlimited"}


def read_file_int(path: str) -> int | None:
    """Read a single integer from a file. Return None on failure."""
    try:
        with open(path) as f:
            text = f.read().strip()
            if text == "max":
                return None
            return int(text)
    except (FileNotFoundError, ValueError, PermissionError):
        return None


def get_container_image() -> str | None:
    container_id = os.environ.get("DOCKER_IMAGE")
    if container_id:
        return container_id

    return None


def detect_hardware() -> dict[str, str | int | float]:
    """Detect CPU/GPU/hardware info.

    When running in Kubernetes container, also logs resource limits
    (cpu_limit, memory_limit_gib) from cgroup files.
    """
    info: dict[str, str | int | float] = {}

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        cap = torch.cuda.get_device_capability(0)
        info["gpu_compute_capability"] = f"{cap[0]}.{cap[1]}"
        info["cuda_version"] = torch.version.cuda or "unknown"
        info["gpu_driver_version"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    else:
        info["gpu_name"] = "none"
        info["gpu_count"] = 0
        info["gpu_compute_capability"] = "n/a"
        info["cuda_version"] = "n/a"
        info["gpu_driver_version"] = "n/a"

    info["os_platform"] = platform.platform()
    info["python_version"] = platform.python_version()

    # ── Container Image (no-op outside container) ───────────────
    container_image = get_container_image()
    if container_image:
        info["container_image"] = container_image

    # ── OMP_NUM_THREADS ──────────────────────────────────────
    omp_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_threads is not None:
        with contextlib.suppress(ValueError):
            info["cpu_requested"] = int(omp_threads)

    # ── CPU/memory limits (K8s) ───────────────────────────────
    info.update(read_ram_limit())

    return info


def detect_image_libraries() -> dict[str, str]:
    info = {}
    try:
        import pyvips

        info["libvips_version"] = (
            f"{pyvips.version(0)}.{pyvips.version(1)}.{pyvips.version(2)}"
        )
    except ImportError:
        info["libvips_version"] = "not_installed"

    try:
        import openslide

        info["openslide_version"] = openslide.__version__
    except ImportError:
        info["openslide_version"] = "not_installed"

    return info


def detect_pytorch() -> dict[str, str]:
    """Detect PyTorch build details for GPU reproducibility."""
    info: dict[str, str] = {}
    info["torch_version"] = torch.__version__

    if torch.cuda.is_available():
        info["cuda_runtime"] = torch.version.cuda or "unknown"
        try:
            info["cudnn_version"] = str(torch.backends.cudnn.version())
        except Exception:
            info["cudnn_version"] = "unknown"
        info["cudnn_enabled"] = str(torch.backends.cudnn.enabled)
    else:
        info["cuda_runtime"] = "n/a"
        info["cudnn_version"] = "n/a"
        info["cudnn_enabled"] = "n/a"

    return info


def detect_seeds() -> dict[str, str]:
    """Read random seed state from environment/config."""
    info: dict[str, str] = {}
    for key in ("SEED", "RANDOM_SEED", "PL_SEED", "TORCH_SEED"):
        val = os.environ.get(key)
        if val is not None:
            try:
                info["seed"] = str(int(val))
                break
            except ValueError:
                pass
    else:
        import random

        info["seed"] = "unset"
        info["random_state"] = str(random.getrandbits(32))

    return info


# ──────────────────────────────────────────────
# Environment snapshot
# ──────────────────────────────────────────────


def snapshot_environment(artifact_dir: str) -> str:
    """Freeze environment to *artifact_dir* and return the pip-freeze text."""
    req_path = os.path.join(artifact_dir, "requirements_frozen.txt")
    with open(req_path, "w") as f:
        import importlib.metadata

        pkgs = []
        for dist in importlib.metadata.distributions():
            name = dist.name or dist.metadata.get("Name")
            version = dist.version
            if name and version:
                pkgs.append(f"{name}=={version}")
        output_text = "\n".join(sorted(pkgs))
        f.write(output_text + "\n")

    for src in ("pyproject.toml", "uv.lock"):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(artifact_dir, src))

    snapshot_system_packages(artifact_dir)

    with open(req_path) as f:
        return f.read()


def snapshot_system_packages(artifact_dir: str) -> None:
    """Snapshot installed system packages for container reproducibility.

    Writes one of:
      - ``system_packages.txt`` (dpkg, Debian/Ubuntu)
    """
    # ── Debian/Ubuntu (dpkg) ─────────────────────────────────
    try:
        dpkg_path = os.path.join(artifact_dir, "system_packages.txt")
        result = subprocess.run(
            ["dpkg-query", "-W", "-f=${Package}=${Version}\n"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            with open(dpkg_path, "w") as f:
                f.write(result.stdout)
            return
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


# ──────────────────────────────────────────────
# Core environment capture (callback-free)
# ──────────────────────────────────────────────


def get_git_tags() -> dict[str, str]:
    """Read git tags from active MLflow run."""
    import mlflow

    git_tags: dict[str, str] = {}
    from mlflow.tracking import MlflowClient

    run = mlflow.active_run()
    if run and run.info and run.info.run_id:
        client = MlflowClient()
        run_data = client.get_run(run.info.run_id)
        git_tags = dict(run_data.data.tags) if run_data.data.tags else {}
    return git_tags


log = logging.getLogger(__name__)


def _attempt(
    section: str,
    call: Callable[[], Any],
    default: Any,
    strict: bool,
) -> Any:
    """Run a detection section, swallowing errors unless *strict*."""
    try:
        return call()
    except Exception as e:
        if strict:
            raise
        log.warning("[capture_environment] %s failed: %s", section, e)
        return default


def _detect_all_sections(
    result: dict[str, Any],
    skip_hardware: bool,
    strict: bool,
) -> None:
    """Fill *result* with all pure-detection sections (no MLflow side effects)."""
    import mlflow

    result["pytorch"] = _attempt("PyTorch detection", detect_pytorch, {}, strict)

    try:
        result["mlflow_version"] = mlflow.__version__
    except Exception:
        result["mlflow_version"] = "unknown"

    def _user() -> tuple[str | None, dict[str, str]]:
        user_run_id, user_tags = lookup_user_run()
        return user_run_id, (user_tags or {})

    user_run_id, user_tags = _attempt("User lookup", _user, (None, {}), strict)
    result.update(user_run_id=user_run_id, user_tags=user_tags)

    result["hardware"] = _attempt(
        "Hardware detection",
        lambda: {} if skip_hardware else detect_hardware(),
        {},
        strict,
    )
    result["seeds"] = _attempt("Seed detection", detect_seeds, {}, strict)
    result["image_libraries"] = _attempt(
        "Image library detection", detect_image_libraries, {}, strict
    )


def _log_env_tags(result: dict[str, Any]) -> None:
    """Set user/dataset/start-time tags on the active MLflow run."""
    import mlflow

    from rationai.mlkit.provenance.dataset import lookup_dataset_run

    env_tags: dict[str, str] = {}
    if result.get("user_run_id"):
        env_tags["user_run_id"] = str(result["user_run_id"])
        utags: dict[str, str] = result.get("user_tags") or {}
        for key in ("username", "real_name", "organization"):
            if key in utags:
                env_tags[key] = utags[key]

    dataset_run_id = lookup_dataset_run()
    if dataset_run_id:
        env_tags["dataset_run_id"] = dataset_run_id

    env_tags["prov_start_time"] = datetime.now(UTC).isoformat()
    mlflow.set_tags(env_tags)


def _log_env_params(result: dict[str, Any]) -> None:
    """Log hardware/pytorch/image-library params and seed tags."""
    import mlflow

    all_params: dict[str, str | float | int] = {
        **result.get("hardware", {}),
        **result.get("pytorch", {}),
        **result.get("image_libraries", {}),
    }
    if all_params:
        mlflow.log_params(all_params)

    seeds = result.get("seeds") or {}
    if seeds:
        mlflow.set_tags({f"seed_{k}": str(v) for k, v in seeds.items()})


def _log_env_snapshot(strict: bool) -> str | None:
    """Freeze the environment and upload it as artifacts."""
    import uuid

    import mlflow

    artifact_dir = f"_mlflow_env_{uuid.uuid4().hex[:8]}"
    try:
        os.makedirs(artifact_dir, exist_ok=True)
        frozen = snapshot_environment(artifact_dir)
        mlflow.log_artifacts(artifact_dir, artifact_path="environment")
        return frozen
    except Exception as e:
        if strict:
            raise
        log.warning("[capture_environment] Environment snapshot failed: %s", e)
        return None
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


def capture_environment(
    skip_hardware: bool = False,
    snapshot_env: bool = True,
    strict: bool = False,
) -> dict[str, object]:
    """Capture full environment metadata and log to active MLflow run.

    This is the core fn used by both EnvironmentCallback and
    @log_environment.  Call directly for script-based logging.

    Returns dict with keys: hardware, frozen_requirements, user_run_id, user_tags.
    """
    import mlflow

    if not mlflow.active_run():
        return {}

    result: dict[str, Any] = {}

    _detect_all_sections(result, skip_hardware, strict)
    _log_env_tags(result)
    _log_env_params(result)

    frozen_requirements = _log_env_snapshot(strict) if snapshot_env else None
    result["frozen_requirements"] = frozen_requirements

    return result


def log_environment(
    skip_hardware: bool = False,
    snapshot_env: bool = True,
    strict: bool = True,
) -> Any:
    """Decorator that captures environment metadata before calling the wrapped function.

    Requires an active MLflow run (set via mlflow.start_run() or @mlflow.autolog()).

    Example::

        @log_environment()
        def train():
            model.fit(X, y)
    """
    from functools import wraps

    def decorator(fn: Any) -> Any:
        @wraps(fn)
        def wrapper(*fn_args: Any, **fn_kwargs: Any) -> Any:
            capture_environment(
                skip_hardware=skip_hardware,
                snapshot_env=snapshot_env,
                strict=strict,
            )
            return fn(*fn_args, **fn_kwargs)

        return wrapper

    return decorator
