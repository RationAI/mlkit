"""Lightning callback that captures PROV-O provenance for MLflow runs.

Slim callback that depends on sibling callbacks for environment and dataset
verification data:

    - :class:`~rationai.mlkit.lightning.callbacks.environment.EnvironmentCallback`
      provides git info, hardware, docker, env snapshot, and user tags.
    - :class:`~rationai.mlkit.lightning.callbacks.dataset_verification.DatasetVerificationCallback`
      provides dataset verification and train/test split results.

When used alone, it falls back to doing its own environment/verification work
so the user gets a single-drop-in experience.

Example::

    from rationai.mlkit.lightning.callbacks import ProvenanceCallback

    trainer = Trainer(
        callbacks=[ProvenanceCallback(model_name="resnet_v1")],
        logger=MLFlowLogger(...),
    )
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
from datetime import UTC, datetime
from typing import Any

import mlflow
import pandas as pd
from lightning.pytorch.callbacks import Callback


log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# OpenProvenance / CPM namespace URIs (§9 — configurable via env var)
# ──────────────────────────────────────────────

_DEFAULT_PROV_PREFIXES = {
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


def _get_prov_prefixes(override: dict[str, str] | None = None) -> dict[str, str]:
    """Return PROV prefix map.

    Priority: explicit override > ``PROV_BASE_URI`` env var > defaults.
    The env var accepts a JSON object of ``{prefix: base_uri}`` pairs that
    merge into (and override) the defaults.
    """
    if override:
        return {**_DEFAULT_PROV_PREFIXES, **override}
    env_json = os.environ.get("PROV_BASE_URI", "")
    if env_json:
        try:
            parsed = json.loads(env_json)
            if not isinstance(parsed, dict):
                raise TypeError("expected a JSON object")
            merged = {**_DEFAULT_PROV_PREFIXES, **parsed}
            return merged
        except (json.JSONDecodeError, TypeError) as e:
            log.warning(f"PROV_BASE_URI is not valid JSON: {e} — using defaults")
    return _DEFAULT_PROV_PREFIXES


_ACTIVITY_HP_KEYS = {
    "learning_rate",
    "lr",
    "batch_size",
    "epochs",
    "optimizer",
    "loss_function",
    "dropout",
    "weight_decay",
    "momentum",
    "num_layers",
    "hidden_size",
    "embedding_dim",
    "num_classes",
    "patch_size",
    "input_size",
    "augmentations",
}

_WSI_PARAM_KEYS = {
    "scanner",
    "slide_id",
    "wsi_id",
    "patient_id",
    "subject_id",
    "institution",
    "site",
    "staining",
    "slicing_method",
}


# ──────────────────────────────────────────────
# Model / Optimizer / Scheduler summaries
# ──────────────────────────────────────────────


def _model_summary(model: Any) -> dict[str, str | int]:
    """Extract architecture details from a torch.nn.Module."""
    info: dict[str, str | int] = {}

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
            f"children={children}",
        )

    layer_summary: str = "\n".join(layer_lines[:20])
    if len(layer_lines) > 20:
        layer_summary += f"\n... ({len(layer_lines)} layers total)"
    info["layer_summary"] = layer_summary

    info["model_class"] = type(model).__name__
    return info


def _optimizer_summary(optimizer: Any) -> dict[str, str | float]:
    """Extract optimizer settings from torch.optim.Optimizer."""
    info: dict[str, str | float] = {}
    info["optimizer_type"] = type(optimizer).__name__
    for name, value in optimizer.defaults.items():
        if isinstance(value, (int, float, bool, str)):
            info[f"opt_{name}"] = value
    return info


def _scheduler_summary(scheduler: Any) -> dict[str, str | float]:
    """Extract scheduler settings."""
    info: dict[str, str | float] = {}
    if scheduler is None:
        info["scheduler_type"] = "none"
        return info

    info["scheduler_type"] = type(scheduler).__name__
    for attr in (
        "step_size",
        "gamma",
        "milestones",
        "factor",
        "patience",
        "min_lr",
        "T_max",
        "eta_min",
    ):
        val = getattr(scheduler, attr, None)
        if val is not None:
            info[f"sch_{attr}"] = (
                str(list(val)) if isinstance(val, (list, tuple)) else str(val)
            )

    if hasattr(scheduler, "optimizer"):
        for name, value in scheduler.optimizer.defaults.items():
            if isinstance(value, (int, float, bool)):
                info[f"sch_opt_{name}"] = value

    return info


# ──────────────────────────────────────────────
# PROV document builder
# ──────────────────────────────────────────────


def _safe_id(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _qualified(prefix: str, local: str) -> str:
    return f"{prefix}:{local}"


def _typed_value(
    value: object, type_prefix: str = "xsd", type_local: str = "string"
) -> list[str]:
    return [str(value)]


def _qualified_name(type_prefix: str, type_local: str) -> dict[str, str]:
    return {"type": "prov:QUALIFIED_NAME", "$": f"{type_prefix}:{type_local}"}


def _iso_timestamp(ts_ms: int | None = None) -> str:
    if ts_ms is not None:
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
    else:
        dt = datetime.now(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000+00:00")


def _build_prov_document(
    run_id: str,
    run_name: str,
    params: dict[str, str],
    metrics: dict[str, float],
    tags: dict[str, str],
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
    split_data: dict[str, object] | None = None,
    requirements: str | None = None,
    verification: dict[str, object] | None = None,
    prov_prefixes: dict[str, str] | None = None,
) -> dict[str, object]:
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

    entities: dict[str, Any] = {}
    activities: dict[str, Any] = {}
    agents: dict[str, Any] = {}
    used: dict[str, Any] = {}
    was_associated_with: dict[str, Any] = {}

    rel_counter = [0]

    def _blank_rel_id() -> str:
        rid = f"_:n{rel_counter[0]}"
        rel_counter[0] += 1
        return rid

    # ── 1. AGENT ───────────────────────────────────────────
    agent_props: dict[str, Any] = {}
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
        params.get("image_path")
        or params.get("wsi_path")
        or params.get("dataset_path")
        or params.get("data_path")
        or params.get("input_path")
    )

    if image_path_candidates:
        wsi_local = _safe_id(f"wsi_{image_path_candidates}")
        wsi_id = _qualified("gen", wsi_local)
        wsi_props: dict[str, Any] = {
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
            "schema:name": _typed_value(
                f"Training dataset ({train_count} train, {test_count} test)"
            ),
            "prov:type": [_qualified_name("sosa", "Sample")],
        }
        used[_blank_rel_id()] = {
            "prov:activity": run_act_id,
            "prov:entity": ds_id,
        }

    # ── 3. RUN ACTIVITY ────────────────────────────────────
    run_activity: dict[str, Any] = {}
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
        if key.startswith(("opt_", "sch_")):
            clean = key.removeprefix("opt_").removeprefix("sch_")
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
    meta_entity: dict[str, Any] = {}
    meta_entity["prov:type"] = [_qualified_name("cpm", "BundleMetadata")]
    org_val = tags.get("organization", "")
    if org_val:
        meta_entity["cpm:organization"] = _typed_value(org_val)

    skip_keys = (
        set(_ACTIVITY_HP_KEYS)
        | _WSI_PARAM_KEYS
        | {
            "image_path",
            "wsi_path",
            "dataset_path",
            "data_path",
            "input_path",
            "segmentation",
            "model",
            "pretrained_model",
            "backbone",
            "feature_extractor",
            "dataset_name",
            "dataset_version",
            "data_split",
            "split",
            "scanner",
            "slide_id",
            "wsi_id",
            "patient_id",
            "subject_id",
            "institution",
            "site",
            "staining",
            "slicing_method",
        }
    )

    for key, val in params.items():
        if key not in skip_keys:
            safe_key = _safe_id(key)
            meta_entity[f"gen:{safe_key}"] = _typed_value(val)

    for key, mval in metrics.items():
        safe_key = _safe_id(key)
        meta_entity[f"gen:{safe_key}"] = _typed_value(mval)

    if split_data:
        meta_entity["gen:split_test_size"] = _typed_value(
            split_data.get("test_size", "0.2")
        )
        meta_entity["gen:split_random_state"] = _typed_value(
            str(split_data.get("random_state", "42"))
        )
        meta_entity["gen:split_stratified"] = ["true"]

        if split_data.get("train"):
            meta_entity["gen:split_train"] = _typed_value(
                json.dumps(split_data["train"])
            )
        if split_data.get("test"):
            meta_entity["gen:split_test"] = _typed_value(
                json.dumps(split_data["test"])
            )

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

    was_generated_by: dict[str, Any] = {}
    was_generated_by[_blank_rel_id()] = {
        "prov:entity": meta_id,
        "prov:activity": run_act_id,
    }

    # ── 5. CPM MAIN ACTIVITY ───────────────────────────────
    main_activity: dict[str, Any] = {}
    main_activity["prov:type"] = [_qualified_name("cpm", "mainActivity")]
    main_activity["cpm:referencedMetaBundleId"] = [
        {"type": "prov:QUALIFIED_NAME", "$": meta_id},
    ]
    main_activity["dct:hasPart"] = [
        {"type": "prov:QUALIFIED_NAME", "$": run_act_id},
    ]
    activities[main_act_id] = main_activity

    # ── 6. RELATIONSHIPS ───────────────────────────────────
    was_associated_with[_blank_rel_id()] = {
        "prov:activity": run_act_id,
        "prov:agent": agent_id,
    }

    # ── 7. ASSEMBLE BUNDLE ─────────────────────────────────
    inner: dict[str, object] = {"prefix": prov_prefixes or _get_prov_prefixes()}
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
# Slim ProvenanceCallback
# ──────────────────────────────────────────────


class ProvenanceCallback(Callback):
    """Lightning callback that captures PROV document + run summary.

    Reads environment data from :class:`EnvironmentCallback` and dataset
    verification/split data from :class:`DatasetVerificationCallback` when
    present as sibling callbacks.  When used alone, falls back to doing its
    own environment/verification work so the user still gets a complete PROV
    document.

    Args:
        model_name: Identifier for this model (shown in run name).
        experiment_name: MLflow experiment name (default: "Training_Pipeline").
        manifest_path: Path to manifest.csv (auto-detected if None).
        data_root: Root directory for dataset files (auto-detected if None).
        test_size: Fraction of data for the test split.
        random_state: Random seed for train/test split.
        fail_fast: Abort training if dataset verification fails.
        strict: If True, re-raise errors from optional provenance steps.
        register_model: If True, auto-log model summary from pl_module.
        register_optimizer: Log optimizer config (or True to auto-detect).
        register_scheduler: Log scheduler config (or True to auto-detect).
        prov_prefixes: Optional override for PROV namespace prefixes.
    """

    def __init__(
        self,
        model_name: str | None = None,
        experiment_name: str = "Training_Pipeline",
        manifest_path: str | None = None,
        data_root: str | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        fail_fast: bool = True,
        strict: bool = False,
        register_model: bool = True,
        register_optimizer: bool = True,
        register_scheduler: bool = True,
        prov_prefixes: dict[str, str] | None = None,
    ) -> None:
        """Initialise the provenance callback.

        Args:
            model_name: Name of the model (defaults to ``MODEL_NAME`` env var or "model").
            experiment_name: MLflow experiment name for the training run.
            manifest_path: Path to manifest.csv (auto-detected if None).
            data_root: Root directory of the dataset (auto-detected if None).
            test_size: Fraction of data for the test split. Set to 0 to skip splitting.
            random_state: Random seed for train/test split.
            fail_fast: Abort training if dataset verification fails.
            strict: If True, re-raise errors from optional provenance steps.
            register_model: If True, auto-log model summary from pl_module.
            register_optimizer: Log optimizer config (or True to auto-detect).
            register_scheduler: Log scheduler config (or True to auto-detect).
            prov_prefixes: Optional override for PROV namespace prefixes.
        """
        self.model_name = model_name or os.environ.get("MODEL_NAME", "model")
        self.experiment_name = experiment_name
        self.manifest_path = manifest_path
        self.data_root = data_root
        self.test_size = test_size
        self.random_state = random_state
        self.fail_fast = fail_fast
        self.strict = strict
        self.register_model = register_model
        self.register_optimizer = register_optimizer
        self.register_scheduler = register_scheduler
        self._prov_prefixes = prov_prefixes

        # Internal state (populated by on_fit_start or sibling callbacks)
        self._run_id: str | None = None
        self._temp_dirs: list[str] = []
        self._split_data: dict[str, object] | None = None
        self._verification: dict[str, object] | None = None
        self._frozen_requirements: str | None = None
        self._git_commit: str = "unknown"
        self._git_url: str = "unknown"
        self._git_branch: str = "unknown"

    # ── helpers ──────────────────────────────────────────────

    def _gather_from_siblings(self, trainer: Any) -> None:
        """Read data already collected by sibling callbacks."""
        from rationai.mlkit.lightning.callbacks.dataset_verification import (
            DatasetVerificationCallback,
        )
        from rationai.mlkit.lightning.callbacks.environment import EnvironmentCallback

        for cb in trainer.callbacks:
            if isinstance(cb, EnvironmentCallback):
                self._git_commit = getattr(cb, "_git_commit", "unknown")
                self._git_url = getattr(cb, "_git_url", "unknown")
                self._git_branch = getattr(cb, "_git_branch", "unknown")
                self._frozen_requirements = getattr(cb, "_frozen_requirements", None)
            elif isinstance(cb, DatasetVerificationCallback):
                self._verification = getattr(cb, "_verification", None)
                self._split_data = getattr(cb, "_split_data", None)

    def _fallback_on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Do environment + verification work when no sibling callbacks exist."""
        from rationai.mlkit.lightning.callbacks.environment import (
            _detect_docker,
            _detect_hardware,
            _lookup_user_run,
            _snapshot_environment,
        )
        from rationai.mlkit.provenance.register_dataset import (
            _detect_manifest,
            _lookup_dataset_run,
            _verify_dataset,
        )

        # ── Git info (read from MLflow tags set by MLFlowLogger) ──
        try:
            run = mlflow.active_run()
            run_tags: dict[str, str] = {}
            if run and run.info and run.info.run_id:
                client = mlflow.tracking.MlflowClient()
                run_data = client.get_run(run.info.run_id)
                run_tags = dict(run_data.data.tags) if run_data.data.tags else {}
            self._git_commit = run_tags.get(
                "mlflow.source.git.commit", run_tags.get("git.commit", "unknown")
            )
            self._git_url = run_tags.get(
                "mlflow.source.git.repoUrl", run_tags.get("git.repo_url", "unknown")
            )
            self._git_branch = run_tags.get(
                "mlflow.source.git.branch", run_tags.get("git.branch", "unknown")
            )
        except Exception as e:
            if self.strict:
                raise
            log.warning("[ProvenanceCallback] Git info failed: %s", e)

        # ── User lookup ─────────────────────────────────────────
        try:
            user_run_id, user_tags = _lookup_user_run()
        except Exception as e:
            if self.strict:
                raise
            log.warning("[ProvenanceCallback] User lookup failed: %s", e)
            user_run_id, user_tags = None, {}

        # ── Hardware (skip if MLflow system metrics are on) ─────
        sys_metrics_on = any(
            getattr(logger, "log_system_metrics", False) for logger in trainer.loggers
        )
        hardware = {} if sys_metrics_on else _detect_hardware()
        docker = _detect_docker()

        # ── Dataset verification & split ────────────────────────
        manifest_path = self.manifest_path
        data_root = self.data_root

        if manifest_path is None:
            manifest_path, data_root = _detect_manifest()
        elif data_root is None:
            data_root = os.path.dirname(os.path.abspath(manifest_path))

        if manifest_path and data_root:
            from rationai.mlkit.provenance.register_dataset import (
                load_manifest,
            )

            # ── Verification (always) ────────────────────────────
            dataset_run_id = _lookup_dataset_run()
            verification = _verify_dataset(manifest_path, data_root, dataset_run_id)
            self._verification = verification or {}
            verification_details: list[str] = verification.get("details", [])

            for detail in verification_details:
                log.info(f"  [ProvenanceCallback] {detail}")

            # Log verification results
            if verification:
                mlflow.log_params(
                    {
                        "dataset_verified": verification["verified"],
                        "dataset_file_sizes_match": verification["file_sizes_match"]
                        is True,
                        "dataset_files_missing": verification["files_missing"],
                        "dataset_files_total": verification["files_total"],
                    }
                )
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
                        + "\n".join(f"  {d}" for d in verification["details"]),
                    )

            # ── Train/test split (only if test_size > 0) ─────────
            if self.test_size > 0:
                from sklearn.model_selection import train_test_split

                samples = load_manifest(manifest_path, data_root)
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
                mlflow.log_params(
                    {
                        "train_samples": len(train_samples),
                        "test_samples": len(test_samples),
                        "train_positive": sum(train_labels),
                        "train_negative": len(train_labels) - sum(train_labels),
                        "test_positive": sum(test_labels),
                        "test_negative": len(test_labels) - sum(test_labels),
                    }
                )
        else:
            log.warning(
                "[ProvenanceCallback] No manifest.csv found — "
                "train/test split not logged."
            )

        # ── Tags ────────────────────────────────────────────────
        tags: dict[str, str] = {}
        if user_run_id:
            tags["user_run_id"] = user_run_id
            for key in ("username", "real_name", "organization"):
                if key in user_tags:
                    tags[key] = user_tags[key]

        dataset_run_id = _lookup_dataset_run()
        if dataset_run_id:
            tags["dataset_run_id"] = dataset_run_id

        tags.update(
            {
                "git_commit": self._git_commit,
                "git_url": self._git_url,
                "git_branch": self._git_branch,
                "prov_start_time": datetime.now(UTC).isoformat(),
            }
        )
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
        except Exception as e:
            if self.strict:
                raise
            log.warning("[ProvenanceCallback] Environment snapshot failed: %s", e)

    # ── lightning hooks ───────────────────────────────────────

    def _ensure_active_run(self, trainer: Any) -> str | None:
        """Ensure MLflow has an active run by triggering the logger's experiment.

        Returns the run_id if successful, or None if no MLFlowLogger is present.
        """
        try:
            from rationai.mlkit.lightning.loggers.mlflow import MLFlowLogger
        except ImportError:
            # Standalone mlflow — rely on whatever active_run exists
            run = mlflow.active_run()
            return run.info.run_id if run else None

        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                # Access .experiment to trigger lazy init + active-run setup
                _ = logger.experiment
                self._run_id = logger.run_id
                return logger.run_id

        run = mlflow.active_run()
        return run.info.run_id if run else None

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Gather environment/verification data from siblings or fall back."""
        from rationai.mlkit.lightning.callbacks.dataset_verification import (
            DatasetVerificationCallback,
        )
        from rationai.mlkit.lightning.callbacks.environment import EnvironmentCallback

        # Ensure the MLFlowLogger has an active run before any fluent API calls
        self._ensure_active_run(trainer)

        # Check if sibling callbacks are present
        has_env = any(isinstance(cb, EnvironmentCallback) for cb in trainer.callbacks)
        has_verify = any(
            isinstance(cb, DatasetVerificationCallback) for cb in trainer.callbacks
        )

        if has_env or has_verify:
            self._gather_from_siblings(trainer)
        else:
            # No siblings — do everything ourselves
            self._fallback_on_fit_start(trainer, pl_module)

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        """Log model/optimizer/scheduler summaries and PROV document."""
        _active_run = mlflow.active_run()
        if _active_run:
            run_id: str = _active_run.info.run_id
        elif self._run_id:
            run_id = self._run_id
        else:
            self._ensure_active_run(trainer)
            if self._run_id:
                run_id = self._run_id
            else:
                return

        # Get a Run object for metadata access
        active_run = mlflow.get_run(run_id)

        # ── Model summary ───────────────────────────────────────
        if self.register_model and pl_module is not None:
            try:
                model_summary = _model_summary(pl_module)
                mlflow.log_params(model_summary)
            except Exception as e:
                if self.strict:
                    raise
                log.warning("[ProvenanceCallback] Model summary failed: %s", e)

        # ── Optimizer summary ───────────────────────────────────
        if self.register_optimizer and pl_module is not None:
            try:
                for opt in trainer.optimizers:
                    optimizer_info = _optimizer_summary(opt)
                    mlflow.log_params(optimizer_info)
                    break
            except Exception as e:
                if self.strict:
                    raise
                log.warning("[ProvenanceCallback] Optimizer summary failed: %s", e)

        # ── Scheduler summary ───────────────────────────────────
        if self.register_scheduler and pl_module is not None:
            try:
                for sched in getattr(trainer, "lr_schedulers", []):
                    scheduler_info = _scheduler_summary(sched.get("scheduler"))
                    mlflow.log_params(scheduler_info)
                    break
            except Exception as e:
                if self.strict:
                    raise
                log.warning("[ProvenanceCallback] Scheduler summary failed: %s", e)

        # ── PROV document + run summary ─────────────────────────
        try:
            run_data = mlflow.get_run(run_id)
            params = {k: str(v) for k, v in run_data.data.params.items()}
            metrics = {k: float(v) for k, v in run_data.data.metrics.items()}
            tags = {
                k: v
                for k, v in run_data.data.tags.items()
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
                    "train_count": len(self._split_data["train"])
                    if isinstance(self._split_data, dict)
                    else 0,  # type: ignore[arg-type]
                    "test_count": len(self._split_data["test"])
                    if isinstance(self._split_data, dict)
                    else 0,  # type: ignore[arg-type]
                    "train": self._split_data["train"] if self._split_data else None,
                    "test": self._split_data["test"] if self._split_data else None,
                }
                if self._split_data
                else None,
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

            # ── PROV document (§9 — configurable prefixes) ──────
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
                }
                if self._split_data
                else None,
                requirements=self._frozen_requirements,
                verification=self._verification,
                prov_prefixes=_get_prov_prefixes(self._prov_prefixes),
            )

            prov_dir = f"_mlflow_prov_{uuid.uuid4().hex[:8]}"
            os.makedirs(prov_dir, exist_ok=True)
            self._temp_dirs.append(prov_dir)
            prov_path = os.path.join(prov_dir, "prov.json")
            with open(prov_path, "w") as f:
                json.dump(prov_doc, f, indent=2)

            mlflow.log_artifact(prov_path, artifact_path="provenance")
            shutil.rmtree(prov_dir, ignore_errors=True)

            log.info("[ProvenanceCallback] Complete → %s", run_id)
        except Exception as e:
            if self.strict:
                raise
            log.warning(
                "[ProvenanceCallback] Could not write provenance artifacts: %s", e
            )

        # ── Clean up temp dirs ──────────────────────────────────
        for d in self._temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
