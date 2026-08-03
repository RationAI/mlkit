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
from typing import Any

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


def _detect_k8s_resources() -> dict[str, float]:
    """Read CPU/memory limits from cgroup (v2 or v1) when on Kubernetes.

    Returns dict with keys: cpu_limit, memory_limit_gb.
    Empty dict if not in container or files unreadable.
    """
    info: dict[str, float] = {}

    # ── Quick heuristic: only bother if K8s indicators present ──
    is_k8s = any(
        k in os.environ for k in ("KUBERNETES_SERVICE_HOST", "KUBERNETES_SERVICE_PORT")
    )
    has_docker_env = os.path.exists("/.dockerenv")
    has_k8s_secrets = os.path.exists("/run/secrets/kubernetes.io")

    if not (is_k8s or has_docker_env or has_k8s_secrets):
        return info

    # ── CPU limit (cgroup v2) ────────────────────────────────
    cpu_limit = _read_cpu_max()
    if cpu_limit is not None:
        info["cpu_limit"] = cpu_limit
    else:
        # ── CPU limit (cgroup v1) ────────────────────────────
        cpu_quota = _read_file_float("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        cpu_period = _read_file_float("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if cpu_quota is not None and cpu_period is not None and cpu_period > 0:
            info["cpu_limit"] = round(cpu_quota / cpu_period, 2)

    # ── Memory limit (cgroup v2) ─────────────────────────────
    mem_max = _read_file_int("/sys/fs/cgroup/memory.max")
    if mem_max is not None and mem_max != 9223372036854771712:
        info["memory_limit_gb"] = round(mem_max / 1e9, 2)
    else:
        # ── Memory limit (cgroup v1) ─────────────────────────
        mem_limit = _read_file_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if mem_limit is not None and mem_limit != 9223372036854771712:
            info["memory_limit_gb"] = round(mem_limit / 1e9, 2)

    return info


def _read_file_int(path: str) -> int | None:
    """Read a single integer from a file. Return None on failure."""
    try:
        with open(path) as f:
            text = f.read().strip()
            if text == "max":
                return None
            return int(text)
    except (FileNotFoundError, ValueError, PermissionError):
        return None


def _read_file_float(path: str) -> float | None:
    """Read a single float from a file. Return None on failure."""
    try:
        with open(path) as f:
            return float(f.read().strip())
    except (FileNotFoundError, ValueError, PermissionError):
        return None


def _read_cpu_max() -> float | None:
    """Parse /sys/fs/cgroup/cpu.max (cgroup v2).

    Format: <limit> <period>  (e.g. '100000 100000' = 1 CPU)
    'max' means unlimited → None.
    """
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            parts = f.read().strip().split()
            if len(parts) != 2 or parts[0] == "max":
                return None
            limit, period = int(parts[0]), int(parts[1])
            if period <= 0:
                return None
            return round(limit / period, 2)
    except (FileNotFoundError, ValueError, PermissionError):
        return None


_HEX64 = set("0123456789abcdef")


def _is_hex64(s: str) -> bool:
    return len(s) == 64 and all(c in _HEX64 for c in s)


def _extract_container_id() -> str | None:
    """Extract container ID from cgroup / mountinfo files.

    Tries multiple sources covering Docker (cgroup v1/v2) and K8s/containerd:
      - /proc/self/cgroup        → docker-<id>.scope, cri-containerd-<id>
      - /proc/self/mountinfo     → /docker/containers/<id>/, standalone <id>
    Returns 64-char hex ID (Docker/containerd) or None outside container.
    """
    # ── Source 1: /proc/self/cgroup ─────────────────────────────
    try:
        with open("/proc/self/cgroup") as f:
            content = f.read()
        import re

        # cgroup v1: docker/<64-hex> or cri-containerd-<64-hex>
        m = re.search(r"cri-containerd-([0-9a-f]{64})", content)
        if m:
            return m.group(1)
        m = re.search(r"docker-([0-9a-f]{64})\\.scope", content)
        if m:
            return m.group(1)

        # cgroup v2 paths may embed the ID in scope names
        for part in content.split("/"):
            if _is_hex64(part):
                return part
    except (FileNotFoundError, PermissionError):
        pass

    # ── Source 2: /proc/self/mountinfo ─────────────────────────
    try:
        with open("/proc/self/mountinfo") as f:
            content = f.read()
        import re

        # Docker: /docker/containers/<64-hex>/resolv.conf ...
        m = re.search(r"/docker/containers/([0-9a-f]{64})/", content)
        if m:
            return m.group(1)

        # containerd/K8s: cri-containerd-<64-hex>.scope in mount paths
        m = re.search(r"cri-containerd-([0-9a-f]{64})", content)
        if m:
            return m.group(1)

        # Fallback: any standalone 64-char hex token
        for part in re.findall(r"[0-9a-f]{64}", content):
            return part
    except (FileNotFoundError, PermissionError):
        pass

    return None


def _detect_hardware() -> dict[str, str | int | float]:
    """Detect CPU/GPU/hardware info.

    When running in Kubernetes container, also logs resource limits
    (cpu_limit, memory_limit_gb) from cgroup files.
    """
    info: dict[str, str | int | float] = {}

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        cap = torch.cuda.get_device_capability(0)
        info["gpu_compute_capability"] = f"{cap[0]}.{cap[1]}"
        info["cuda_version"] = torch.version.cuda or "unknown"
    else:
        info["gpu_name"] = "none"

    info["os_platform"] = platform.platform()
    info["python_version"] = platform.python_version()

    # ── K8s resources (no-op outside container) ──────────────
    k8s_resources = _detect_k8s_resources()
    info.update(k8s_resources)

    # ── Container ID (no-op outside container) ───────────────
    container_id = _extract_container_id()
    if container_id:
        info["container_id"] = container_id

    return info


def _detect_pytorch() -> dict[str, str]:
    """Detect PyTorch build details for GPU reproducibility."""
    info: dict[str, str] = {}
    info["torch_version"] = torch.__version__
    info["torch_git_version"] = torch.version.git_version or "unknown"

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


def _detect_seeds() -> dict[str, str]:
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
# Docker detection
# ──────────────────────────────────────────────


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

    _snapshot_system_packages(artifact_dir)

    with open(req_path) as f:
        return f.read()


def _snapshot_system_packages(artifact_dir: str) -> None:
    """Snapshot installed system packages for container reproducibility.

    Writes one of:
      - ``system_packages.txt`` (dpkg, Debian/Ubuntu)
      - ``system_packages_rpm.txt`` (rpm, RHEL/Fedora)
      - ``system_packages_apk.txt`` (apk, Alpine)
    """
    # ── Debian/Ubuntu (dpkg) ─────────────────────────────────
    try:
        dpkg_path = os.path.join(artifact_dir, "system_packages.txt")
        result = subprocess.run(
            ["dpkg", "--get-selections"],
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

    # ── RHEL/Fedora (rpm) ────────────────────────────────────
    try:
        rpm_path = os.path.join(artifact_dir, "system_packages_rpm.txt")
        result = subprocess.run(
            ["rpm", "-qa", "--qf", "%{NAME}-%{VERSION}-%{RELEASE}\n"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            with open(rpm_path, "w") as f:
                f.write(result.stdout)
            return
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # ── Alpine (apk) ─────────────────────────────────────────
    try:
        apk_path = os.path.join(artifact_dir, "system_packages_apk.txt")
        result = subprocess.run(
            ["apk", "list", "--installed"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            with open(apk_path, "w") as f:
                f.write(result.stdout)
            return
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


# ──────────────────────────────────────────────
# Core environment capture (callback-free)
# ──────────────────────────────────────────────


def _get_git_tags() -> dict[str, str]:
    """Read git tags from active MLflow run."""
    import mlflow

    git_tags: dict[str, str] = {}
    run = mlflow.active_run()
    if run and run.info and run.info.run_id:
        client = mlflow.tracking.MlflowClient()
        run_data = client.get_run(run.info.run_id)
        git_tags = dict(run_data.data.tags) if run_data.data.tags else {}
    return git_tags


def capture_environment(
    skip_hardware: bool = False,
    snapshot_env: bool = True,
    strict: bool = False,
) -> dict[str, object]:
    """Capture full environment metadata and log to active MLflow run.

    This is the core fn used by both EnvironmentCallback and
    @log_environment.  Call directly for script-based logging.

    Returns dict with keys: git_commit, git_url, git_branch, hardware,
    frozen_requirements, user_run_id, user_tags.
    """
    import mlflow

    from rationai.mlkit.provenance.dataset import _lookup_dataset_run

    if not mlflow.active_run():
        return {}

    result: dict[str, object] = {}

    # ── PyTorch build info ───────────────────────────────────
    try:
        result["pytorch"] = _detect_pytorch()
    except Exception as e:
        if strict:
            raise
        log = logging.getLogger(__name__)
        log.warning("[capture_environment] PyTorch detection failed: %s", e)
        result["pytorch"] = {}

    # ── MLflow version ───────────────────────────────────────
    try:
        result["mlflow_version"] = mlflow.__version__
    except Exception:
        result["mlflow_version"] = "unknown"

    # ── Git info ──────────────────────────────────────────────
    try:
        git_tags = _get_git_tags()
        result["git_commit"] = git_tags.get(
            "mlflow.source.git.commit", git_tags.get("git.commit", "unknown")
        )
        result["git_url"] = git_tags.get(
            "mlflow.source.git.repoUrl", git_tags.get("git.repo_url", "unknown")
        )
        result["git_branch"] = git_tags.get(
            "mlflow.source.git.branch", git_tags.get("git.branch", "unknown")
        )
    except Exception as e:
        if strict:
            raise
        log = logging.getLogger(__name__)
        log.warning("[capture_environment] Git info failed: %s", e)
        result.update(git_commit="unknown", git_url="unknown", git_branch="unknown")

    # ── User lookup ───────────────────────────────────────────
    try:
        user_run_id, user_tags = _lookup_user_run()
        result["user_run_id"] = user_run_id
        result["user_tags"] = user_tags or {}
    except Exception as e:
        if strict:
            raise
        log = logging.getLogger(__name__)
        log.warning("[capture_environment] User lookup failed: %s", e)
        result.update(user_run_id=None, user_tags={})

    # ── Hardware ──────────────────────────────────────────────
    try:
        if not skip_hardware:
            result["hardware"] = _detect_hardware()
        else:
            result["hardware"] = {}
    except Exception as e:
        if strict:
            raise
        log = logging.getLogger(__name__)
        log.warning("[capture_environment] Hardware detection failed: %s", e)
        result["hardware"] = {}

    # ── Seeds ─────────────────────────────────────────────────
    try:
        result["seeds"] = _detect_seeds()
    except Exception as e:
        if strict:
            raise
        log = logging.getLogger(__name__)
        log.warning("[capture_environment] Seed detection failed: %s", e)
        result["seeds"] = {}

    # ── Log tags to MLflow ────────────────────────────────────
    env_tags: dict[str, str] = {}
    if result.get("user_run_id"):
        env_tags["user_run_id"] = str(result["user_run_id"])
        utags: dict[str, str] = result.get("user_tags") or {}  # type: ignore[assignment]
        for key in ("username", "real_name", "organization"):
            if key in utags:
                env_tags[key] = utags[key]

    dataset_run_id = _lookup_dataset_run()
    if dataset_run_id:
        env_tags["dataset_run_id"] = dataset_run_id

    env_tags.update(
        {
            "git_commit": str(result.get("git_commit", "unknown")),
            "git_url": str(result.get("git_url", "unknown")),
            "git_branch": str(result.get("git_branch", "unknown")),
            "prov_start_time": datetime.now(UTC).isoformat(),
        }
    )
    mlflow.set_tags(env_tags)

    # ── Log params ────────────────────────────────────────────
    all_params: dict[str, str | float | int] = {
        **result.get("hardware", {}),  # type: ignore
        **result.get("pytorch", {}),  # type: ignore
    }
    if all_params:
        mlflow.log_params(all_params)

    # ── Log seeds as tags (strings) ───────────────────────────
    seeds = result.get("seeds") or {}  # type: ignore
    if seeds:
        mlflow.set_tags({f"seed_{k}": str(v) for k, v in seeds.items()})

    # ── Environment snapshot ──────────────────────────────────
    frozen_requirements = None
    if snapshot_env:
        import uuid

        artifact_dir = f"_mlflow_env_{uuid.uuid4().hex[:8]}"
        os.makedirs(artifact_dir, exist_ok=True)
        try:
            frozen_requirements = _snapshot_environment(artifact_dir)
            mlflow.log_artifacts(artifact_dir, artifact_path="environment")
        except Exception as e:
            if strict:
                raise
            log = logging.getLogger(__name__)
            log.warning("[capture_environment] Environment snapshot failed: %s", e)
    result["frozen_requirements"] = frozen_requirements

    return result


def log_environment(
    skip_hardware: bool = False,
    snapshot_env: bool = True,
    strict: bool = False,
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
