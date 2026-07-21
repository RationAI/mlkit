"""Lightning callback that captures full PROV-O provenance for MLflow runs.

Replaces the plain-PyTorch ``@autolog`` decorator from
``rationai.mlkit.provenance.provenance`` with a Lightning-native callback
that hooks into ``on_fit_start`` / ``on_fit_end``.

Example::

    from rationai.mlkit.lightning.callbacks import ProvenanceCallback

    trainer = Trainer(
        callbacks=[ProvenanceCallback(model_name="resnet_v1")],
        logger=MLFlowLogger(...),
    )
"""

from __future__ import annotations

import os
import re
import json
import uuid
import shutil
import platform
import hashlib
import subprocess
from datetime import datetime, timezone

import mlflow
import torch
import pandas as pd
from lightning.pytorch.callbacks import Callback


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

_ACTIVITY_HP_KEYS = {
    "learning_rate", "lr", "batch_size", "epochs", "optimizer",
    "loss_function", "dropout", "weight_decay", "momentum",
    "num_layers", "hidden_size", "embedding_dim", "num_classes",
    "patch_size", "input_size", "augmentations",
}

_WSI_PARAM_KEYS = {
    "scanner", "slide_id", "wsi_id", "patient_id", "subject_id",
    "institution", "site", "staining", "slicing_method",
}


# ──────────────────────────────────────────────
# Helpers
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


def _lookup_user_run():
    """Find the user run from User_Registry.  Auto-detect username."""
    from rationai.mlkit.provenance.register_dataset import _lookup_experiment

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


def _snapshot_environment(artifact_dir):
    """Freeze environment to *artifact_dir* and return the pip-freeze text."""
    req_path = os.path.join(artifact_dir, "requirements_frozen.txt")
    with open(req_path, "w") as f:
        subprocess.run(["uv", "pip", "freeze"], stdout=f, check=True)

    for src in ("pyproject.toml", "uv.lock"):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(artifact_dir, src))

    with open(req_path) as f:
        return f.read()


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
# PROV document builder
# ──────────────────────────────────────────────

def _safe_id(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def _qualified(prefix: str, local: str) -> str:
    return f"{prefix}:{local}"


def _typed_value(value, type_prefix="xsd", type_local="string") -> list:
    return [str(value)]


def _qualified_name(type_prefix: str, type_local: str) -> dict:
    return {"type": "prov:QUALIFIED_NAME", "$": f"{type_prefix}:{type_local}"}


def _iso_timestamp(ts_ms: int | None = None) -> str:
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
    """Build an OpenProvenance-compatible PROV document dict."""

    username = tags.get("username", tags.get("mlflow.user", "unknown"))
    agent_local = _safe_id(f"user_{username}")
    agent_id = _qualified("gen", agent_local)

    run_act_local = _safe_id(f"run_{run_id}")
    run_act_id = _qualified("gen", run_act_local)

    meta_local = run_id
    meta_id = _qualified("meta", meta_local)

    main_act_local = f"TrainingRun_{run_id[:8]}"
    main_act_id = _qualified("blank", main_act_local)

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

    # ── 1. AGENT ───────────────────────────────────────────
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

    # ── 2. INPUT ENTITIES ──────────────────────────────────
    image_path_candidates = (
        params.get("image_path") or
        params.get("wsi_path") or
        params.get("dataset_path") or
        params.get("data_path") or
        params.get("input_path")
    )

    if image_path_candidates:
        wsi_local = _safe_id(f"wsi_{image_path_candidates}")
        wsi_id = _qualified("gen", wsi_local)
        wsi_props: dict[str, list] = {
            "schema:name": _typed_value(f"Input: {image_path_candidates}"),
            "prov:type": [_qualified_name("sosa", "Sample")],
        }
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

    # ── 3. RUN ACTIVITY ────────────────────────────────────
    run_activity: dict[str, object] = {}
    run_activity["prov:type"] = [_qualified_name("schema", "Action")]
    run_activity["prov:startTime"] = [_iso_timestamp(start_time_ms)]
    run_activity["prov:endTime"] = [_iso_timestamp(end_time_ms)]
    run_activity["schema:name"] = _typed_value(run_name)

    exp_name = params.get("model_name", "")
    if exp_name:
        run_activity["gen:experiment_name"] = _typed_value(exp_name)

    if "model_class" in params:
        run_activity["gen:model_config"] = _typed_value(params["model_class"])

    git_commit = tags.get("git_commit", tags.get("mlflow.source.git.commit", ""))
    if git_commit:
        run_activity["schema:identifier"] = _typed_value(git_commit)

    for key in ("pretrained_model", "backbone", "feature_extractor"):
        if key in params:
            run_activity["gen:pretrained_model"] = _typed_value(params[key])

    for key, prov_key in [
        ("dataset_name", "gen:dataset_name"),
        ("dataset_version", "gen:dataset_version"),
        ("data_split", "gen:data_split"),
        ("split", "gen:data_split"),
    ]:
        if key in params:
            run_activity[prov_key] = _typed_value(params[key])

    for key in _ACTIVITY_HP_KEYS:
        if key in params:
            run_activity[f"gen:{key}"] = _typed_value(params[key])

    for key, val in params.items():
        if key.startswith("opt_") or key.startswith("sch_"):
            clean = key
            while clean.startswith("opt_") or clean.startswith("sch_"):
                if clean.startswith("opt_"):
                    clean = clean[4:]
                elif clean.startswith("sch_"):
                    clean = clean[4:]
            if f"gen:{clean}" not in run_activity:
                run_activity[f"gen:{clean}"] = _typed_value(val)

    for tag_key, prov_key in [
        ("mlflow.gpu.count", "gen:gpu_count"),
        ("mlflow.gpu.names", "gen:gpu_names"),
        ("mlflow.cpu.count", "gen:cpu_count"),
        ("mlflow.memory_gb", "gen:memory_gb"),
    ]:
        if tag_key in tags:
            run_activity[prov_key] = _typed_value(tags[tag_key])

    for param_key, prov_key in [
        ("gpu_count", "gen:gpu_count"),
        ("gpu_name", "gen:gpu_names"),
        ("cpu_count_logical", "gen:cpu_count"),
        ("ram_total_gb", "gen:memory_gb"),
    ]:
        if param_key in params and prov_key not in run_activity:
            run_activity[prov_key] = _typed_value(params[param_key])

    git_url = tags.get("git_url", tags.get("mlflow.source.git.remote", ""))
    if git_url:
        run_activity["gen:git_remote"] = _typed_value(git_url)

    source_name = tags.get("mlflow.source.name", "")
    if source_name:
        run_activity["gen:source_name"] = _typed_value(source_name)

    for key, prov_key in [
        ("segmentation", "gen:segmentation_config"),
        ("model", "gen:model_config"),
    ]:
        if key in params:
            run_activity[prov_key] = _typed_value(params[key])

    activities[run_act_id] = run_activity

    # ── 4. CPM METADATA ENTITY ─────────────────────────────
    meta_entity: dict[str, list] = {}
    meta_entity["prov:type"] = [_qualified_name("cpm", "BundleMetadata")]
    org_val = tags.get("organization", "")
    if org_val:
        meta_entity["cpm:organization"] = _typed_value(org_val)

    skip_keys = set(_ACTIVITY_HP_KEYS) | _WSI_PARAM_KEYS | {
        "image_path", "wsi_path", "dataset_path", "data_path", "input_path",
        "segmentation", "model", "pretrained_model", "backbone", "feature_extractor",
        "dataset_name", "dataset_version", "data_split", "split",
        "scanner", "slide_id", "wsi_id", "patient_id", "subject_id",
        "institution", "site", "staining", "slicing_method",
    }

    for key, val in params.items():
        if key not in skip_keys:
            safe_key = _safe_id(key)
            meta_entity[f"gen:{safe_key}"] = _typed_value(val)

    for key, val in metrics.items():
        safe_key = _safe_id(key)
        meta_entity[f"gen:{safe_key}"] = _typed_value(val)

    if split_data:
        meta_entity["gen:split_test_size"] = _typed_value(split_data.get("test_size", "0.2"))
        meta_entity["gen:split_random_state"] = _typed_value(str(split_data.get("random_state", "42")))
        meta_entity["gen:split_stratified"] = ["true"]

        if split_data.get("train"):
            meta_entity["gen:split_train"] = [_typed_value(json.dumps(split_data["train"]))[0]]
        if split_data.get("test"):
            meta_entity["gen:split_test"] = [_typed_value(json.dumps(split_data["test"]))[0]]

    if requirements:
        meta_entity["gen:requirements"] = [requirements]

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

    was_generated_by: dict[str, dict] = {}
    was_generated_by[_blank_rel_id()] = {
        "prov:entity": meta_id,
        "prov:activity": run_act_id,
    }

    # ── 5. CPM MAIN ACTIVITY ───────────────────────────────
    main_activity: dict[str, object] = {}
    main_activity["prov:type"] = [_qualified_name("cpm", "mainActivity")]
    main_activity["cpm:referencedMetaBundleId"] = [
        {"type": "prov:QUALIFIED_NAME", "$": meta_id}
    ]
    main_activity["dct:hasPart"] = [
        {"type": "prov:QUALIFIED_NAME", "$": run_act_id}
    ]
    activities[main_act_id] = main_activity

    # ── 6. RELATIONSHIPS ───────────────────────────────────
    was_associated_with[_blank_rel_id()] = {
        "prov:activity": run_act_id,
        "prov:agent": agent_id,
    }

    # ── 7. ASSEMBLE BUNDLE ─────────────────────────────────
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
# ProvenanceCallback
# ──────────────────────────────────────────────

class ProvenanceCallback(Callback):
    """Lightning callback that captures full PROV-O provenance.

    Logs user tags, git info, hardware, docker detection, environment
    snapshot, dataset verification, train/test split, model summary,
    optimizer/scheduler summary, console output capture, and builds
    a self-contained PROV document on training completion.

    Args:
        model_name: Identifier for this model (shown in run name).
        experiment_name: MLflow experiment name (default: "Training_Pipeline").
        manifest_path: Path to manifest.csv (auto-detected if None).
        data_root: Root directory for dataset files (auto-detected if None).
        test_size: Fraction of data for the test split.
        random_state: Random seed for train/test split.
        log_stream: Whether to capture stdout/stderr into console.log.
        fail_fast: Abort training if dataset verification fails.
        register_model: If True, auto-log model summary from pl_module.
        register_optimizer: Pass an optimizer to log its config (or True to auto-detect).
        register_scheduler: Pass a scheduler to log its config (or True to auto-detect).
    """

    def __init__(
        self,
        model_name: str | None = None,
        experiment_name: str = "Training_Pipeline",
        manifest_path: str | None = None,
        data_root: str | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        log_stream: bool = True,
        fail_fast: bool = True,
        register_model: bool = True,
        register_optimizer: bool = True,
        register_scheduler: bool = True,
    ):
        self.model_name = model_name or os.environ.get("MODEL_NAME", "model")
        self.experiment_name = experiment_name
        self.manifest_path = manifest_path
        self.data_root = data_root
        self.test_size = test_size
        self.random_state = random_state
        self.log_stream = log_stream
        self.fail_fast = fail_fast
        self.register_model = register_model
        self.register_optimizer = register_optimizer
        self.register_scheduler = register_scheduler

        # Internal state
        self._run_id: str | None = None
        self._mlflow_run = None
        self._temp_dirs: list[str] = []
        self._split_data: dict | None = None
        self._verification: dict | None = None
        self._frozen_requirements: str | None = None
        self._optimizer_info: dict | None = None
        self._scheduler_info: dict | None = None
        self._git_commit: str = "unknown"
        self._git_url: str = "unknown"
        self._git_branch: str = "unknown"

    def on_fit_start(self, trainer, pl_module):  # noqa: ARG002
        """Run all provenance setup at the start of training."""
        from rationai.mlkit.provenance.register_dataset import (
            _detect_manifest,
            _lookup_dataset_run,
            _verify_dataset,
        )

        # ── Auto-detect everything ──────────────────────────────
        user_run_id, user_tags = _lookup_user_run()
        git_commit, git_url, git_branch = _get_git_info()
        self._git_commit = git_commit
        self._git_url = git_url
        self._git_branch = git_branch

        hardware = _detect_hardware()
        docker = _detect_docker()

        # ── Dataset detection & split ───────────────────────────
        manifest_path = self.manifest_path
        data_root = self.data_root

        if manifest_path is None:
            manifest_path, data_root = _detect_manifest()

        if manifest_path and data_root:
            dataset_run_id = _lookup_dataset_run(manifest_path)
            verification = _verify_dataset(manifest_path, data_root, dataset_run_id)
            self._verification = verification

            for detail in verification["details"]:
                print(f"  [ProvenanceCallback] {detail}")

            # Train/test split
            from sklearn.model_selection import train_test_split
            df = pd.read_csv(manifest_path)
            samples = []
            for _, row in df.iterrows():
                rel = row["wsi_path"]
                full = os.path.join(data_root, rel) if not os.path.isabs(rel) else rel
                samples.append({"path": full, "label": int(row["cancer"])})

            train_samples, test_samples = train_test_split(
                samples,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=[s["label"] for s in samples],
            )

            self._split_data = {
                "train": train_samples,
                "test": test_samples,
                "test_size": self.test_size,
                "random_state": self.random_state,
            }

            # Log split as artifact
            split_dir = f"_mlflow_split_{uuid.uuid4().hex[:8]}"
            os.makedirs(split_dir, exist_ok=True)
            self._temp_dirs.append(split_dir)
            for subset_name, subset_samples in [
                ("train", train_samples),
                ("test", test_samples),
            ]:
                split_file = os.path.join(split_dir, f"{subset_name}_split.csv")
                pd.DataFrame(subset_samples).to_csv(split_file, index=False)

            mlflow.log_artifacts(split_dir, artifact_path="split")

            # Log split counts
            train_labels = [s["label"] for s in train_samples]
            test_labels = [s["label"] for s in test_samples]
            mlflow.log_params({
                "train_samples": len(train_samples),
                "test_samples": len(test_samples),
                "train_positive": sum(train_labels),
                "train_negative": len(train_labels) - sum(train_labels),
                "test_positive": sum(test_labels),
                "test_negative": len(test_labels) - sum(test_labels),
            })

            # Log verification results
            mlflow.log_params({
                "dataset_verified": verification["verified"],
                "dataset_file_sizes_match": verification["file_sizes_match"] is True,
                "dataset_files_missing": verification["files_missing"],
                "dataset_files_total": verification["files_total"],
            })
            if verification["verified"]:
                mlflow.set_tag("dataset_verification", "VERIFIED")
            else:
                mlflow.set_tag("dataset_verification", "MISMATCH")
                mlflow.set_tag(
                    "dataset_verification_details",
                    "; ".join(verification["details"]),
                )

            if self.fail_fast and not verification["verified"]:
                raise RuntimeError(
                    "Dataset verification failed — aborting training.\n"
                    + "\n".join(f"  {d}" for d in verification["details"])
                )
        else:
            print("[ProvenanceCallback] WARNING: No manifest.csv found — "
                  "train/test split not logged.")

        # ── Tags ────────────────────────────────────────────────
        tags: dict[str, str] = {}
        if user_run_id:
            tags["user_run_id"] = user_run_id
            for key in ("username", "real_name", "organization"):
                if key in user_tags:
                    tags[key] = user_tags[key]

        if dataset_run_id := _lookup_dataset_run(manifest_path):
            tags["dataset_run_id"] = dataset_run_id

        tags.update({
            "git_commit": git_commit,
            "git_url": git_url,
            "git_branch": git_branch,
            "prov_start_time": datetime.now(timezone.utc).isoformat(),
        })
        mlflow.set_tags(tags)

        # ── Params: hardware + docker + split config ────────────
        all_params: dict[str, str | float | int] = {
            "model_name": self.model_name,
            **hardware,
            **docker,
            "split_test_size": self.test_size,
            "split_random_state": self.random_state,
            "split_stratified": True,
        }
        mlflow.log_params(all_params)

        # ── Environment snapshot ────────────────────────────────
        artifact_dir = f"_mlflow_env_{uuid.uuid4().hex[:8]}"
        os.makedirs(artifact_dir, exist_ok=True)
        self._temp_dirs.append(artifact_dir)
        try:
            self._frozen_requirements = _snapshot_environment(artifact_dir)
            mlflow.log_artifacts(artifact_dir, artifact_path="environment")
        except Exception:
            pass

    def on_fit_end(self, trainer, pl_module):  # noqa: ARG002
        """Log model/optimizer/scheduler summaries and PROV document."""
        active_run = mlflow.active_run()
        if not active_run:
            return
        run_id = active_run.info.run_id

        # ── Model summary ───────────────────────────────────────
        if self.register_model and pl_module is not None:
            try:
                model_summary = _model_summary(pl_module)
                mlflow.log_params(model_summary)
            except Exception:
                pass

        # ── Optimizer summary ───────────────────────────────────
        if self.register_optimizer and pl_module is not None:
            try:
                for opt in trainer.optimizers:
                    self._optimizer_info = _optimizer_summary(opt)
                    mlflow.log_params(self._optimizer_info)
                    break
            except Exception:
                pass

        # ── Scheduler summary ───────────────────────────────────
        if self.register_scheduler and pl_module is not None:
            try:
                for sched in trainer.lr_schedulers:
                    self._scheduler_info = _scheduler_summary(sched.get("scheduler"))
                    mlflow.log_params(self._scheduler_info)
                    break
            except Exception:
                pass

        # ── PROV document + run summary ─────────────────────────
        try:
            run_data = mlflow.get_run(run_id)
            params = {k: str(v) for k, v in run_data.data.params.items()}
            metrics = {k: float(v) for k, v in run_data.data.metrics.items()}
            tags = {
                k: v for k, v in run_data.data.tags.items()
                if not k.startswith("mlflow.")
            }

            # ── Run summary JSON ────────────────────────────────
            summary_dir = f"_mlflow_summary_{uuid.uuid4().hex[:8]}"
            os.makedirs(summary_dir, exist_ok=True)
            self._temp_dirs.append(summary_dir)
            summary_path = os.path.join(summary_dir, "run_summary.json")

            summary = {
                "model_name": self.model_name,
                "params": dict(run_data.data.params),
                "metrics": {k: float(v) for k, v in run_data.data.metrics.items()},
                "tags": tags,
                "run_id": run_id,
                "experiment_name": self.experiment_name,
                "split": {
                    "test_size": self.test_size,
                    "random_state": self.random_state,
                    "stratified": True,
                    "train_count": len(self._split_data["train"]) if self._split_data else 0,
                    "test_count": len(self._split_data["test"]) if self._split_data else 0,
                    "train": self._split_data["train"] if self._split_data else None,
                    "test": self._split_data["test"] if self._split_data else None,
                } if self._split_data else None,
                "dataset_verification": self._verification,
                "requirements": self._frozen_requirements,
                "source": {
                    "git_commit": self._git_commit,
                    "git_branch": self._git_branch,
                    "git_remote": self._git_url,
                },
            }

            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            mlflow.log_artifact(summary_path, artifact_path="provenance")
            shutil.rmtree(summary_dir, ignore_errors=True)

            # ── PROV document ───────────────────────────────────
            prov_doc = _build_prov_document(
                run_id=run_id,
                run_name=active_run.info.run_name or f"Training_{self.model_name}",
                params=params,
                metrics=metrics,
                tags=tags,
                start_time_ms=active_run.info.start_time,
                end_time_ms=active_run.info.end_time,
                split_data={
                    "test_size": self.test_size,
                    "random_state": self.random_state,
                    "train": self._split_data["train"] if self._split_data else None,
                    "test": self._split_data["test"] if self._split_data else None,
                } if self._split_data else None,
                requirements=self._frozen_requirements,
                verification=self._verification,
            )

            prov_dir = f"_mlflow_prov_{uuid.uuid4().hex[:8]}"
            os.makedirs(prov_dir, exist_ok=True)
            self._temp_dirs.append(prov_dir)
            prov_path = os.path.join(prov_dir, "prov.json")
            with open(prov_path, "w") as f:
                json.dump(prov_doc, f, indent=2)

            mlflow.log_artifact(prov_path, artifact_path="provenance")
            shutil.rmtree(prov_dir, ignore_errors=True)

            print(f"\n[ProvenanceCallback] Complete → {run_id}")
        except Exception as e:
            print(f"[ProvenanceCallback] WARNING: Could not write provenance artifacts: {e}")

        # ── Clean up temp dirs ──────────────────────────────────
        for d in self._temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
