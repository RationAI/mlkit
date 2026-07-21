"""
Automatic PROV-O-aware provenance logger for MLflow.

Inspired by rationai.mlkit.autolog — the user decorates their training function
and all metadata is captured automatically.

Usage:

    from provenance import autolog

    @autolog(model_name="resnet_baseline_v1")
    def train(run):
        run.log_params({"learning_rate": 1e-3})
        model = build_model()
        run.register_model(model)
        # ... training loop ...
        run.save_model(model)

Everything else (user, dataset, hardware, docker, git, environment,
train/test split, console output) is detected and logged automatically.
"""

from __future__ import annotations

import io
import os
import re
import json
import uuid
import shutil
import types
import platform
import hashlib
import subprocess
import contextlib
from datetime import datetime, timezone
from functools import partial, wraps
from collections.abc import Callable

import mlflow
import torch
import pandas as pd

# ──────────────────────────────────────────────
# OpenProvenance / CPM namespace URIs
# ──────────────────────────────────────────────

_PROV_PREFIXES = {
    "storage": "http://localhost:8083/api/v1/documents/",
    "meta": "http://localhost:8083/api/v1/documents/meta/",
    "schema": "https://schema.org/",
    "cpm": "https://www.commonprovenancemodel.org/cpm-namespace-v1-0/",
    "blank": "https://openprovenance.org/blank/",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "gen": "gen/",
    "dct": "http://purl.org/dc/terms/",
    "prov": "http://www.w3.org/ns/prov#",
    "sosa": "http://www.w3.org/ns/sosa/",
}

# Hyperparameter keys that go on the activity vs. metadata entity
_ACTIVITY_HP_KEYS = {
    "learning_rate", "lr", "batch_size", "epochs", "optimizer",
    "loss_function", "dropout", "weight_decay", "momentum",
    "num_layers", "hidden_size", "embedding_dim", "num_classes",
    "patch_size", "input_size", "augmentations",
}

# Param keys that map to WSI entity properties
_WSI_PARAM_KEYS = {
    "scanner", "slide_id", "wsi_id", "patient_id", "subject_id",
    "institution", "site", "staining", "slicing_method",
}


# ──────────────────────────────────────────────
# Auto-detection helpers
# ──────────────────────────────────────────────

def _get_git_info():
    """Return (commit, remote_url, branch) or ('unknown', ...) on failure."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        remote = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except subprocess.CalledProcessError:
        commit, remote, branch = "unknown", "unknown", "unknown"
    return commit, remote, branch


# ── Dataset helpers live in register_dataset.py ──────────────────────
from .register_dataset import (  # noqa: F401
    _lookup_dataset_run,
    _lookup_experiment,
    _verify_dataset,
)


def _lookup_user_run():
    """Find the user run from User_Registry.  Auto-detect username."""
    username = os.environ.get("MLFLOW_USER")
    if not username:
        try:
            username = subprocess.check_output(
                ["git", "config", "user.name"], stderr=subprocess.DEVNULL,
            ).decode().strip()
        except subprocess.CalledProcessError:
            pass
    if not username:
        username = os.environ.get("USER", "unknown")

    exp_id = _lookup_experiment("User_Registry")
    if exp_id is None:
        return None, {}

    runs = mlflow.search_runs(experiment_ids=[exp_id])
    if runs.empty:
        return None, {}

    matched = runs[runs["tags.username"] == username]
    if matched.empty:
        matched = runs.head(1)

    row = matched.iloc[0]
    run = mlflow.get_run(row.run_id)
    return row.run_id, dict(run.data.tags)


# ── Dataset verification lives in register_dataset.py ─────────────────


def _detect_hardware():
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


def _detect_docker():
    info: dict[str, str | bool] = {"docker": False}

    if os.path.exists("/.dockerenv"):
        info["docker"] = True

    # Fallback: cgroup v1 / v2
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
                        if len(p) == 64 and all(
                            c in "0123456789abcdef" for c in p
                        ):
                            info["container_id_short"] = p[:12]
                            break
        except FileNotFoundError:
            pass

    # Try docker inspect if socket is available inside container
    if info["docker"]:
        cid = info.get("container_id_short", "")
        if cid:
            try:
                result = subprocess.run(
                    ["docker", "inspect", "--format={{.Config.Image}}", cid],
                    capture_output=True, text=True, timeout=5,
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


def _detect_manifest():
    """Walk data/ looking for manifest.csv."""
    for root_dir in ("data", "test_data", "."):
        for dirpath, _, filenames in os.walk(root_dir):
            if "manifest.csv" in filenames:
                return (
                    os.path.join(dirpath, "manifest.csv"),
                    os.path.dirname(os.path.abspath(
                        os.path.join(dirpath, "manifest.csv")
                    )),
                )
    return None, None


def _snapshot_environment(artifact_dir):
    """Freeze environment to *artifact_dir* and return the pip-freeze text."""
    req_path = os.path.join(artifact_dir, "requirements_frozen.txt")
    with open(req_path, "w") as f:
        subprocess.run(["uv", "pip", "freeze"], stdout=f, check=True)

    for src in ("pyproject.toml", "uv.lock"):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(artifact_dir, src))

    # Return the frozen requirements text for embedding into provenance
    with open(req_path) as f:
        return f.read()


def _source_hash(source_file: str) -> str | None:
    """Return SHA-256 of a source file (for reproducibility verification)."""
    try:
        h = hashlib.sha256()
        with open(source_file, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except (FileNotFoundError, PermissionError):
        return None


def _prepare_split(manifest_path, data_root, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(manifest_path)
    samples = []
    for _, row in df.iterrows():
        rel = row["wsi_path"]
        full = os.path.join(data_root, rel) if not os.path.isabs(rel) else rel
        samples.append({"path": full, "label": int(row["cancer"])})

    train_s, test_s = train_test_split(
        samples,
        test_size=test_size,
        random_state=random_state,
        stratify=[s["label"] for s in samples],
    )
    return list(train_s), list(test_s)


# ──────────────────────────────────────────────
# Model / optimizer / scheduler introspection
# ──────────────────────────────────────────────

def _model_summary(model):
    """Extract architecture details from a torch.nn.Module."""
    info: dict[str, str | int] = {}

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    info["total_parameters"] = total_params
    info["trainable_parameters"] = trainable_params

    layer_lines = []
    for name, module in model.named_modules():
        if name == "":
            continue
        param_count = sum(p.numel() for p in module.parameters(recurse=False))
        children = len(list(module.children()))
        layer_lines.append(
            f"{name}({type(module).__name__}): params={param_count}, "
            f"children={children}"
        )

    info["layer_summary"] = "\n".join(layer_lines[:20])
    if len(layer_lines) > 20:
        info["layer_summary"] += f"\n... ({len(layer_lines)} layers total)"

    info["model_class"] = type(model).__name__
    return info


def _optimizer_summary(optimizer):
    """Extract optimizer settings from torch.optim.Optimizer."""
    info: dict[str, str | float] = {}
    info["optimizer_type"] = type(optimizer).__name__
    for name, value in optimizer.defaults.items():
        if isinstance(value, (int, float, bool, str)):
            info[f"opt_{name}"] = value
    return info


def _scheduler_summary(scheduler):
    """Extract scheduler settings."""
    info: dict[str, str | float] = {}
    if scheduler is None:
        info["scheduler_type"] = "none"
        return info

    info["scheduler_type"] = type(scheduler).__name__
    for attr in ("step_size", "gamma", "milestones", "factor",
                 "patience", "min_lr", "T_max", "eta_min"):
        val = getattr(scheduler, attr, None)
        if val is not None:
            info[f"sch_{attr}"] = list(val) if isinstance(val, (list, tuple)) else val

    if hasattr(scheduler, "optimizer"):
        for name, value in scheduler.optimizer.defaults.items():
            if isinstance(value, (int, float, bool)):
                info[f"sch_opt_{name}"] = value

    return info


# ──────────────────────────────────────────────
# OpenProvenance / CPM PROV document builder
# ──────────────────────────────────────────────

def _safe_id(name: str) -> str:
    """Sanitise a string for use as a PROV identifier fragment."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def _qualified(prefix: str, local: str) -> str:
    """Return a qualified name like 'gen:run_abc123'."""
    return f"{prefix}:{local}"


def _typed_value(value, type_prefix="xsd", type_local="string") -> list:
    """Wrap a string value as [value] — matching Java's array convention."""
    return [str(value)]


def _qualified_name(type_prefix: str, type_local: str) -> dict:
    """Build a prov:QUALIFIED_NAME type descriptor."""
    return {"type": "prov:QUALIFIED_NAME", "$": f"{type_prefix}:{type_local}"}


def _iso_timestamp(ts_ms: int | None = None) -> str:
    """Return an ISO-8601 timestamp string (ms since epoch or now)."""
    if ts_ms is not None:
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    else:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000+00:00")


def _build_prov_document(
    run_id: str,
    run_name: str,
    params: dict[str, str],
    metrics: dict[str, float],
    tags: dict[str, str],
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
    split_data: dict | None = None,
    requirements: str | None = None,
    verification: dict | None = None,
) -> dict:
    """Build an OpenProvenance-compatible PROV document dict.

    Structure mirrors the Java prov_mlflow output:
      - bundle wrapper with storage:<run_id> key
      - prefix namespace declarations
      - entity, activity, agent sections
      - wasAssociatedWith, used relationship sections
    """

    # ── Derive identifiers ────────────────────────────────
    username = tags.get("username", tags.get("mlflow.user", "unknown"))
    agent_local = _safe_id(f"user_{username}")
    agent_id = _qualified("gen", agent_local)

    run_act_local = _safe_id(f"run_{run_id}")
    run_act_id = _qualified("gen", run_act_local)

    meta_local = run_id
    meta_id = _qualified("meta", meta_local)

    main_act_local = f"TrainingRun_{run_id[:8]}"
    main_act_id = _qualified("blank", main_act_local)

    # ── Collect sections ──────────────────────────────────
    entities: dict[str, dict] = {}
    activities: dict[str, dict] = {}
    agents: dict[str, dict] = {}
    used: dict[str, dict] = {}
    was_associated_with: dict[str, dict] = {}

    rel_counter = [0]

    def _blank_rel_id() -> str:
        rid = f"_:n{rel_counter[0]}"
        rel_counter[0] += 1
        return rid

    # ── 1. AGENT (researcher) ─────────────────────────────
    agent_props: dict[str, list] = {}
    real_name = tags.get("real_name", username)
    agent_props["schema:name"] = _typed_value(real_name)
    email = tags.get("mlflow.source.git.user.email", f"{username}@unknown")
    agent_props["schema:email"] = _typed_value(email)
    org = tags.get("organization", "")
    if org:
        agent_props["schema:affiliation"] = _typed_value(org)
    agent_props["prov:type"] = [_qualified_name("schema", "Person")]
    agents[agent_id] = agent_props

    # ── 2. INPUT ENTITIES (WSI / dataset samples) ─────────
    # Collect unique sample paths from params or tags
    sample_paths: list[str] = []
    for key in ("train_samples", "test_samples"):
        if key in params:
            pass  # counts, not paths — skip

    # Try to find WSI-related params
    image_path_candidates = (
        params.get("image_path") or
        params.get("wsi_path") or
        params.get("dataset_path") or
        params.get("data_path") or
        params.get("input_path")
    )

    # If we have a manifest reference, create a single dataset entity
    if image_path_candidates:
        wsi_local = _safe_id(f"wsi_{image_path_candidates}")
        wsi_id = _qualified("gen", wsi_local)
        wsi_props: dict[str, list] = {
            "schema:name": _typed_value(f"Input: {image_path_candidates}"),
            "prov:type": [_qualified_name("sosa", "Sample")],
        }
        # Add optional WSI metadata from params
        if "scanner" in params:
            wsi_props["gen:scanner"] = _typed_value(params["scanner"])
        for pk, prov_key in [
            ("slide_id", "schema:identifier"),
            ("wsi_id", "schema:identifier"),
            ("patient_id", "gen:patient_pseudonym"),
            ("subject_id", "gen:patient_pseudonym"),
            ("institution", "gen:origin_institution"),
            ("site", "gen:origin_institution"),
            ("staining", "gen:staining_method"),
            ("slicing_method", "gen:slicing_method"),
        ]:
            if pk in params:
                wsi_props[prov_key] = _typed_value(params[pk])

        entities[wsi_id] = wsi_props
        used[_blank_rel_id()] = {
            "prov:activity": run_act_id,
            "prov:entity": wsi_id,
        }
    else:
        # Fallback: create a generic dataset entity from split info
        train_count = params.get("train_samples", "0")
        test_count = params.get("test_samples", "0")
        ds_local = _safe_id(f"dataset_{run_id[:8]}")
        ds_id = _qualified("gen", ds_local)
        entities[ds_id] = {
            "schema:name": _typed_value(f"Training dataset ({train_count} train, {test_count} test)"),
            "prov:type": [_qualified_name("sosa", "Sample")],
        }
        used[_blank_rel_id()] = {
            "prov:activity": run_act_id,
            "prov:entity": ds_id,
        }

    # ── 3. RUN ACTIVITY (the ML training) ─────────────────
    run_activity: dict[str, object] = {}
    run_activity["prov:type"] = [_qualified_name("schema", "Action")]
    run_activity["prov:startTime"] = [_iso_timestamp(start_time_ms)]
    run_activity["prov:endTime"] = [_iso_timestamp(end_time_ms)]
    run_activity["schema:name"] = _typed_value(run_name)

    # Experiment name from tags
    exp_name = params.get("model_name", "")
    if exp_name:
        run_activity["gen:experiment_name"] = _typed_value(exp_name)

    # Model config
    if "model_class" in params:
        run_activity["gen:model_config"] = _typed_value(params["model_class"])

    # Git commit
    git_commit = tags.get("git_commit", tags.get("mlflow.source.git.commit", ""))
    if git_commit:
        run_activity["schema:identifier"] = _typed_value(git_commit)

    # Backward-compatible model params
    for key in ("pretrained_model", "backbone", "feature_extractor"):
        if key in params:
            run_activity["gen:pretrained_model"] = _typed_value(params[key])

    # Dataset info
    for key, prov_key in [
        ("dataset_name", "gen:dataset_name"),
        ("dataset_version", "gen:dataset_version"),
        ("data_split", "gen:data_split"),
        ("split", "gen:data_split"),
    ]:
        if key in params:
            run_activity[prov_key] = _typed_value(params[key])

    # Hyperparameters
    for key in _ACTIVITY_HP_KEYS:
        if key in params:
            run_activity[f"gen:{key}"] = _typed_value(params[key])

    # Also add opt_ and sch_ prefixed params as hyperparams (stripped)
    # Strip ALL leading prefixes to avoid sch_opt_lr → gen:opt_lr pollution
    for key, val in params.items():
        if key.startswith("opt_") or key.startswith("sch_"):
            clean = key
            while clean.startswith("opt_") or clean.startswith("sch_"):
                if clean.startswith("opt_"):
                    clean = clean[4:]
                elif clean.startswith("sch_"):
                    clean = clean[4:]
            if f"gen:{clean}" not in run_activity:  # avoid duplicates
                run_activity[f"gen:{clean}"] = _typed_value(val)

    # Hardware from tags (mlflow.* convention) and our custom params
    for tag_key, prov_key in [
        ("mlflow.gpu.count", "gen:gpu_count"),
        ("mlflow.gpu.names", "gen:gpu_names"),
        ("mlflow.cpu.count", "gen:cpu_count"),
        ("mlflow.memory_gb", "gen:memory_gb"),
    ]:
        if tag_key in tags:
            run_activity[prov_key] = _typed_value(tags[tag_key])

    # Our custom hardware params
    for param_key, prov_key in [
        ("gpu_count", "gen:gpu_count"),
        ("gpu_name", "gen:gpu_names"),
        ("cpu_count_logical", "gen:cpu_count"),
        ("ram_total_gb", "gen:memory_gb"),
    ]:
        if param_key in params and prov_key not in run_activity:
            run_activity[prov_key] = _typed_value(params[param_key])

    # Git remote / source
    git_url = tags.get("git_url", tags.get("mlflow.source.git.remote", ""))
    if git_url:
        run_activity["gen:git_remote"] = _typed_value(git_url)

    source_name = tags.get("mlflow.source.name", "")
    if source_name:
        run_activity["gen:source_name"] = _typed_value(source_name)

    # Segmentation / model params (if present)
    for key, prov_key in [
        ("segmentation", "gen:segmentation_config"),
        ("model", "gen:model_config"),
    ]:
        if key in params:
            run_activity[prov_key] = _typed_value(params[key])

    activities[run_act_id] = run_activity

    # ── 4. CPM METADATA ENTITY ────────────────────────────
    meta_entity: dict[str, list] = {}
    meta_entity["prov:type"] = [_qualified_name("cpm", "BundleMetadata")]
    org_val = tags.get("organization", "")
    if org_val:
        meta_entity["cpm:organization"] = _typed_value(org_val)

    # Skip keys already placed on the activity
    skip_keys = set(_ACTIVITY_HP_KEYS) | _WSI_PARAM_KEYS | {
        "image_path", "wsi_path", "dataset_path", "data_path", "input_path",
        "segmentation", "model", "pretrained_model", "backbone", "feature_extractor",
        "dataset_name", "dataset_version", "data_split", "split",
        "scanner", "slide_id", "wsi_id", "patient_id", "subject_id",
        "institution", "site", "staining", "slicing_method",
    }

    # Remaining params → metadata entity
    for key, val in params.items():
        if key not in skip_keys:
            safe_key = _safe_id(key)
            meta_entity[f"gen:{safe_key}"] = _typed_value(val)

    # Metrics → metadata entity
    for key, val in metrics.items():
        safe_key = _safe_id(key)
        meta_entity[f"gen:{safe_key}"] = _typed_value(val)

    # ── Reproducibility: embedded dataset splits ───────────
    if split_data:
        # Embed as JSON string so the PROV document is self-contained
        meta_entity["gen:split_test_size"] = _typed_value(split_data.get("test_size", "0.2"))
        meta_entity["gen:split_random_state"] = _typed_value(str(split_data.get("random_state", "42")))
        meta_entity["gen:split_stratified"] = ["true"]

        if split_data.get("train"):
            meta_entity["gen:split_train"] = [_typed_value(json.dumps(split_data["train"]))[0]]
        if split_data.get("test"):
            meta_entity["gen:split_test"] = [_typed_value(json.dumps(split_data["test"]))[0]]

    # ── Reproducibility: frozen requirements ───────────────
    if requirements:
        meta_entity["gen:requirements"] = [requirements]

    # ── Reproducibility: dataset verification ──────────────
    if verification:
        meta_entity["gen:dataset_verified"] = [str(verification.get("verified", False))]
        meta_entity["gen:dataset_run_id"] = [verification.get("dataset_run_id", "")]
        fsm = verification.get("file_sizes_match")
        if fsm is not None:
            meta_entity["gen:file_sizes_match"] = [str(fsm)]
        fm = verification.get("files_missing", 0)
        ft = verification.get("files_total", 0)
        meta_entity["gen:files_missing"] = [str(fm)]
        meta_entity["gen:files_total"] = [str(ft)]

    # Selected tags → metadata entity
    for tag_key in (
        "mlflow.source.git.branch",
        "mlflow.source.git.repo_url",
        "mlflow.parentRunId",
        "mlflow.note.content",
    ):
        if tag_key in tags:
            safe_key = _safe_id(tag_key)
            meta_entity[f"gen:{safe_key}"] = _typed_value(tags[tag_key])

    entities[meta_id] = meta_entity

    # wasGeneratedBy: meta entity ← run activity
    was_generated_by: dict[str, dict] = {}
    was_generated_by[_blank_rel_id()] = {
        "prov:entity": meta_id,
        "prov:activity": run_act_id,
    }

    # ── 5. CPM MAIN ACTIVITY ──────────────────────────────
    main_activity: dict[str, object] = {}
    main_activity["prov:type"] = [_qualified_name("cpm", "mainActivity")]
    main_activity["cpm:referencedMetaBundleId"] = [
        {"type": "prov:QUALIFIED_NAME", "$": meta_id}
    ]
    main_activity["dct:hasPart"] = [
        {"type": "prov:QUALIFIED_NAME", "$": run_act_id}
    ]
    activities[main_act_id] = main_activity

    # ── 6. RELATIONSHIPS ──────────────────────────────────
    was_associated_with[_blank_rel_id()] = {
        "prov:activity": run_act_id,
        "prov:agent": agent_id,
    }

    # ── 7. ASSEMBLE BUNDLE ────────────────────────────────
    inner: dict[str, object] = {"prefix": _PROV_PREFIXES}
    if entities:
        inner["entity"] = entities
    if activities:
        inner["activity"] = activities
    if agents:
        inner["agent"] = agents
    if was_associated_with:
        inner["wasAssociatedWith"] = was_associated_with
    if was_generated_by:
        inner["wasGeneratedBy"] = was_generated_by
    if used:
        inner["used"] = used

    bundle_key = f"storage:{run_id}"
    return {"bundle": {bundle_key: inner}}


# ──────────────────────────────────────────────
# Console stream capture  (like mlkit.stream)
# ──────────────────────────────────────────────

_CONSOLE_LOG_NAME = "console.log"


class _StreamCapture:
    """Captures stdout/stderr and writes them to MLflow as console.log."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._buffer = io.StringIO()
        self._originals = {}

    def __enter__(self):
        import sys
        for stream_name in ("stdout", "stderr"):
            stream = getattr(sys, stream_name)
            original_write = stream.write
            self._originals[stream_name] = original_write

            def _make_wrapper(buf, orig):
                def wrapper(text):
                    buf.write(text)
                    return orig(text)
                return wrapper

            setattr(stream, "write", _make_wrapper(self._buffer, self._originals[stream_name]))

    def __exit__(self, *args):
        import sys
        for stream_name in ("stdout", "stderr"):
            original = self._originals.get(stream_name)
            if original is not None:
                setattr(getattr(sys, stream_name), "write", original)

        text = self._buffer.getvalue()
        if text.strip():
            try:
                client = mlflow.tracking.MlflowClient()
                log_path = os.path.join(
                    f"_mlflow_console_{uuid.uuid4().hex[:8]}",
                    _CONSOLE_LOG_NAME,
                )
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "w") as f:
                    f.write(text)
                client.log_artifact(self.run_id, log_path, artifact_path="logs")
                shutil.rmtree(os.path.dirname(log_path), ignore_errors=True)
            except Exception:
                pass  # don't fail the run


# ──────────────────────────────────────────────
# Run helper object (injected into the user function)
# ──────────────────────────────────────────────

class _Run:
    """Handle passed to the user's training function.

    Provides logging methods and model/optimizer/scheduler registration.
    """

    def __init__(self, run_id: str):
        self._run_id = run_id
        self._model = None
        self._optimizer_info: dict | None = None
        self._scheduler_info: dict | None = None
        self.train_paths: list[str] = []
        self.test_paths: list[str] = []
        self.train_labels: list[int] = []
        self.test_labels: list[int] = []
        self._split_data: dict | None = None  # {"train": [...], "test": [...]}

    def _set_split(self, train_samples: list[dict], test_samples: list[dict]):
        """Store the full split data for embedding into provenance."""
        self._split_data = {"train": train_samples, "test": test_samples}

    # ── Logging (forwarded to mlflow) ─────────

    def log_param(self, key, value):
        mlflow.log_param(key, value)

    def log_params(self, params_dict):
        mlflow.log_params(params_dict)

    def log_metric(self, key, value, step=None):
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics_dict, step=None):
        mlflow.log_metrics(metrics_dict, step=step)

    def log_artifact(self, local_path, artifact_path=None):
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_artifacts(self, local_dir, artifact_path=None):
        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)

    # ── Registration (logged at the end) ───────

    def register_model(self, model):
        self._model = model

    def register_optimizer(self, optimizer):
        self._optimizer_info = _optimizer_summary(optimizer)

    def register_scheduler(self, scheduler):
        self._scheduler_info = _scheduler_summary(scheduler)

    # ── Model saving ──────────────────────────

    def save_model(self, model, name="model", **kwargs):
        if "export_model" not in kwargs:
            kwargs["export_model"] = False
        mlflow.pytorch.log_model(model, name, **kwargs)


# ──────────────────────────────────────────────
# Decorator — the main entry point
# ──────────────────────────────────────────────

def autolog(
    model_name: str | None = None,
    experiment_name: str = "Training_Pipeline",
    test_size: float = 0.2,
    random_state: int = 42,
    log_stream: bool = True,
    fail_fast: bool = True,
):
    """Decorator for automatic provenance logging.

    All metadata (user, dataset, hardware, docker, git, environment,
    train/test split, model architecture, optimizer, scheduler, console
    output) is captured automatically.

    Args:
        model_name: Identifier for this model (shown in run name).
                     Defaults to MODEL_NAME env var or "model".
        experiment_name: MLflow experiment name.
        test_size: Fraction of data for the test split.
        random_state: Random seed for train/test split.
        log_stream: Whether to capture stdout/stderr into console.log.
        fail_fast: If True (default), abort the run with RuntimeError
                   when dataset verification fails.

    Example:
        from provenance import autolog

        @autolog(model_name="resnet_baseline_v1")
        def train(run):
            run.log_params({"learning_rate": 1e-3})
            model = build_model()
            run.register_model(model)
            optimizer = optim.SGD(model.parameters(), lr=1e-3)
            run.register_optimizer(optimizer)
            for epoch in range(50):
                loss = train_epoch(...)
                run.log_metrics({"train_loss": loss}, step=epoch)
            run.save_model(model)

        if __name__ == "__main__":
            train()
    """

    def decorator(func: Callable[..., None]) -> Callable[[], None]:
        @wraps(func)
        def wrapper():
            _run_autolog(
                func=func,
                model_name=model_name or os.environ.get("MODEL_NAME", "model"),
                experiment_name=experiment_name,
                test_size=test_size,
                random_state=random_state,
                log_stream=log_stream,
                fail_fast=fail_fast,
            )

        return wrapper

    return decorator


def _run_autolog(func, model_name, experiment_name, test_size, random_state, log_stream, fail_fast=True):
    """Core autolog logic."""
    _temp_dirs: list[str] = []

    # ── Auto-detect everything ──────────────────────────────
    user_run_id, user_tags = _lookup_user_run()
    manifest_path, data_root = _detect_manifest()
    dataset_run_id = _lookup_dataset_run(manifest_path)
    git_commit, git_url, git_branch = _get_git_info()
    hardware = _detect_hardware()
    docker = _detect_docker()

    # ── Start MLflow run ───────────────────────────────────
    mlflow.set_experiment(experiment_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Training_{model_name}_{ts}"

    mlflow_run = mlflow.start_run(run_name=run_name)
    run_id = mlflow_run.info.run_id

    # ── 1. Tags (PROV relationships) ───────────────────────
    tags: dict[str, str] = {}
    if user_run_id:
        tags["user_run_id"] = user_run_id
        for key in ("username", "real_name", "organization"):
            if key in user_tags:
                tags[key] = user_tags[key]

    if dataset_run_id:
        tags["dataset_run_id"] = dataset_run_id

    tags.update({
        "git_commit": git_commit,
        "git_url": git_url,
        "git_branch": git_branch,
        "prov_start_time": datetime.now(timezone.utc).isoformat(),
    })
    mlflow.set_tags(tags)

    # ── 2. Params: hardware + docker + split config ────────
    all_params: dict[str, str | float | int] = {
        "model_name": model_name,
        **hardware,
        **docker,
        "split_test_size": test_size,
        "split_random_state": random_state,
        "split_stratified": True,
    }
    mlflow.log_params(all_params)

    # ── 3. Environment snapshot ────────────────────────────
    artifact_dir = f"_mlflow_env_{uuid.uuid4().hex[:8]}"
    os.makedirs(artifact_dir, exist_ok=True)
    _temp_dirs.append(artifact_dir)
    frozen_requirements: str | None = None
    try:
        frozen_requirements = _snapshot_environment(artifact_dir)
        mlflow.log_artifacts(artifact_dir, artifact_path="environment")
    except Exception:
        pass

    # ── 4. Train/test split from manifest ──────────────────
    run_handle = _Run(run_id)

    if manifest_path and data_root:
        train_samples, test_samples = _prepare_split(
            manifest_path, data_root, test_size, random_state
        )

        run_handle.train_paths = [s["path"] for s in train_samples]
        run_handle.test_paths = [s["path"] for s in test_samples]
        run_handle.train_labels = [s["label"] for s in train_samples]
        run_handle.test_labels = [s["label"] for s in test_samples]

        # Store full split data for embedding into provenance
        run_handle._set_split(train_samples, test_samples)

        mlflow.log_params({
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "train_positive": sum(run_handle.train_labels),
            "train_negative": len(run_handle.train_labels) - sum(run_handle.train_labels),
            "test_positive": sum(run_handle.test_labels),
            "test_negative": len(run_handle.test_labels) - sum(run_handle.test_labels),
        })

        split_dir = f"_mlflow_split_{uuid.uuid4().hex[:8]}"
        os.makedirs(split_dir, exist_ok=True)
        _temp_dirs.append(split_dir)
        for subset_name, subset_samples in [
            ("train", train_samples),
            ("test", test_samples),
        ]:
            split_file = os.path.join(split_dir, f"{subset_name}_split.csv")
            pd.DataFrame(subset_samples).to_csv(split_file, index=False)

        mlflow.log_artifacts(split_dir, artifact_path="split")

        # ── 4b. Verify dataset against Dataset_Registry ────
        verification = _verify_dataset(manifest_path, data_root, dataset_run_id)
        for detail in verification["details"]:
            print(f"  [autolog] {detail}")

        # Log verification results as params + tags
        mlflow.log_params({
            "dataset_verified": verification["verified"],
            "dataset_file_sizes_match": verification["file_sizes_match"] is True,
            "dataset_files_missing": verification["files_missing"],
            "dataset_files_total": verification["files_total"],
        })
        if verification["verified"]:
            tags["dataset_verification"] = "VERIFIED"
        else:
            tags["dataset_verification"] = "MISMATCH"
            tags["dataset_verification_details"] = "; ".join(verification["details"])
        mlflow.set_tags(tags)

        # ── Hard abort on mismatch ─────────────────────
        if fail_fast and not verification["verified"]:
            mlflow.end_run(status='FAILED')
            raise RuntimeError(
                "Dataset verification failed — aborting training.\n"
                + "  ".join("  " + d for d in verification["details"])
            )

    else:
        print("[autolog] WARNING: No manifest.csv found — "
              "train/test split not logged.")
        verification = None

    # ── 5. Run the user's training function ────────────────
    try:
        if log_stream:
            with _StreamCapture(run_id):
                func(run_handle)
        else:
            func(run_handle)
    except Exception:
        # Still log what we can before re-raising
        raise
    finally:
        # ── Log model/optimizer/scheduler (if registered) ──
        if run_handle._model is not None:
            mlflow.log_params(_model_summary(run_handle._model))

        if run_handle._optimizer_info is not None:
            mlflow.log_params(run_handle._optimizer_info)
        if run_handle._scheduler_info is not None:
            mlflow.log_params(run_handle._scheduler_info)

        # ── Write self-contained provenance JSON ───────────
        # Embeds splits + requirements so the run is reproducible
        # without needing MLflow at all.
        try:
            run_data = mlflow.get_run(run_id)
            summary_dir = f"_mlflow_summary_{uuid.uuid4().hex[:8]}"
            os.makedirs(summary_dir, exist_ok=True)
            summary_path = os.path.join(summary_dir, "run_summary.json")

            # Source file hash for verification
            source_file = getattr(func, "__wrapped__", func).__code__.co_filename
            source_hash = _source_hash(source_file) if source_file else None

            summary = {
                "model_name": model_name,
                "params": dict(run_data.data.params),
                "metrics": {k: float(v) for k, v in run_data.data.metrics.items()},
                "tags": {
                    k: v for k, v in run_data.data.tags.items()
                    if not k.startswith("mlflow.")
                },
                "run_id": run_id,
                "run_name": run_name,
                "experiment_name": experiment_name,

                # ── Reproducibility: dataset splits ─────────
                "split": {
                    "test_size": test_size,
                    "random_state": random_state,
                    "stratified": True,
                    "train_count": len(run_handle.train_paths),
                    "test_count": len(run_handle.test_paths),
                    "train": run_handle._split_data["train"] if run_handle._split_data else None,
                    "test": run_handle._split_data["test"] if run_handle._split_data else None,
                } if run_handle._split_data else None,

                # ── Reproducibility: dataset verification ───
                "dataset_verification": verification if verification is not None else None,

                # ── Reproducibility: frozen environment ─────
                "requirements": frozen_requirements,

                # ── Source verification ─────────────────────
                "source": {
                    "file": source_file,
                    "sha256": source_hash,
                    "git_commit": git_commit,
                    "git_branch": git_branch,
                    "git_remote": git_url,
                },
            }

            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            mlflow.log_artifact(summary_path, artifact_path="provenance")
            shutil.rmtree(summary_dir, ignore_errors=True)

            # ── Build & log PROV document ───────────────────
            prov_doc = _build_prov_document(
                run_id=run_id,
                run_name=run_name,
                params={k: str(v) for k, v in run_data.data.params.items()},
                metrics={k: float(v) for k, v in run_data.data.metrics.items()},
                tags={
                    k: v for k, v in run_data.data.tags.items()
                    if not k.startswith("mlflow.")
                },
                start_time_ms=mlflow_run.info.start_time,
                end_time_ms=mlflow_run.info.end_time,
                split_data={
                    "test_size": test_size,
                    "random_state": random_state,
                    "train": run_handle._split_data["train"] if run_handle._split_data else None,
                    "test": run_handle._split_data["test"] if run_handle._split_data else None,
                } if run_handle._split_data else None,
                requirements=frozen_requirements,
                verification=verification,
            )

            prov_dir = f"_mlflow_prov_{uuid.uuid4().hex[:8]}"
            os.makedirs(prov_dir, exist_ok=True)
            _temp_dirs.append(prov_dir)
            prov_path = os.path.join(prov_dir, "prov.json")
            with open(prov_path, "w") as f:
                json.dump(prov_doc, f, indent=2)

            mlflow.log_artifact(prov_path, artifact_path="provenance")
            shutil.rmtree(prov_dir, ignore_errors=True)

            print(f"\n[autolog] Complete → {run_id}")
        except Exception as e:
            print(f"[autolog] WARNING: Could not write provenance artifacts: {e}")

        # ── Clean up temp dirs ─────────────────────────────
        for d in _temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
