"""Provenance logging for dataset split operations.

Logs split statistics (train/test/folds), source dataset reference,
environment snapshot, and PROV-O document to the active MLflow run.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from omegaconf import DictConfig

from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.mlkit.provenance.common import (
    _iso_timestamp,
    _qualified,
    _qualified_name,
    _safe_id,
    get_prov_prefixes,
)
from rationai.mlkit.provenance.environment import capture_environment


def log_split_provenance(
    splits: dict[str, pd.DataFrame],
    logger: MLFlowLogger,
    config: DictConfig,
    *,
    dataset_name: str = "ulcerative-colitis-dysplasia",
    version: str = "1.0.0",
    snapshot_env: bool = True,
) -> dict[str, Any]:
    """Log split metadata + PROV document to active MLflow run.

    Args:
        splits: Dict mapping split name → DataFrame (e.g. {"train": ..., "test_preliminary": ..., "test_final": ...}).
        logger: MLFlowLogger instance for the current run.
        config: Hydra config (must have ``splits`` and optionally ``n_folds``, ``random_state``).
        dataset_name: Human-readable dataset name.
        version: Dataset version string.
        snapshot_env: Whether to capture environment snapshot.

    Returns:
        Dict with split stats and PROV document.
    """
    # ── Environment ────────────────────────────────────────
    if snapshot_env:
        capture_environment(snapshot_env=True)

    # ── Collect split stats ────────────────────────────────
    split_stats: dict[str, Any] = {}
    for name, df in splits.items():
        stats: dict[str, Any] = {
            "num_samples": len(df),
            "num_cases": df["case_id"].nunique() if "case_id" in df.columns else 0,
        }
        if "fold" in df.columns:
            stats["num_folds"] = (
                (df["fold"] >= 0).sum() if (df["fold"] >= 0).any() else 0
            )
            stats["folds"] = sorted(df["fold"].unique().tolist())
        split_stats[name] = stats

    # ── Log params + tags ──────────────────────────────────
    total_samples = sum(s["num_samples"] for s in split_stats.values())
    total_cases = sum(s["num_cases"] for s in split_stats.values())

    _n_folds = getattr(config, "n_folds", 0)
    _random_state = getattr(config, "random_state", 42)

    mlflow.log_params(
        {
            "total_samples": total_samples,
            "total_cases": total_cases,
            "split_train_size": config.splits.get("train", 0),
            "split_test_preliminary_size": config.splits.get("test_preliminary", 0),
            "split_test_final_size": config.splits.get("test_final", 0),
            "n_folds": _n_folds,
            "random_state": _random_state,
        }
    )

    for name, stats in split_stats.items():
        prefix = f"split_{name}_"
        mlflow.log_params(
            {
                f"{prefix}samples": stats["num_samples"],
                f"{prefix}cases": stats["num_cases"],
            }
        )

    mlflow.set_tags(
        {
            "dataset_name": dataset_name,
            "version": version,
            "split_stats": json.dumps(split_stats),
        }
    )

    # ── PROV document ──────────────────────────────────────
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("No active MLflow run — call inside @autolog")
    run_id = active_run.info.run_id

    prov_doc = _build_split_prov(
        run_id=run_id,
        dataset_name=dataset_name,
        version=version,
        split_stats=split_stats,
        n_folds=_n_folds,
        random_state=_random_state,
    )

    # ── Upload artifacts (PROV JSON) ───────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        prov_dir = Path(tmpdir) / "provenance"
        prov_dir.mkdir(exist_ok=True)

        prov_path = prov_dir / "prov.json"
        prov_path.write_text(json.dumps(prov_doc, indent=2), encoding="utf-8")

        logger.log_artifact(str(prov_path), artifact_path="provenance")

    return {
        "prov_doc": prov_doc,
        "split_stats": split_stats,
        "total_samples": total_samples,
        "total_cases": total_cases,
    }


def _build_split_prov(
    run_id: str,
    dataset_name: str,
    version: str,
    split_stats: dict[str, dict[str, Any]],
    n_folds: int,
    random_state: int,
    prov_prefixes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a PROV document for a dataset split operation."""
    prefixes = prov_prefixes or get_prov_prefixes()

    username = _get_username()

    # ── AGENT ──────────────────────────────────────────────
    agent_local = _safe_id(f"user_{username}")
    agent_id = _qualified("gen", agent_local)
    agent_props: dict[str, list[Any]] = {}
    agent_props["schema:name"] = [_typed_value_str(username)]
    agent_props["prov:type"] = [_qualified_name("schema", "Person")]

    # ── IDs ────────────────────────────────────────────────
    run_act_local = _safe_id(f"run_{run_id}")
    run_act_id = _qualified("gen", run_act_local)

    ds_local = _safe_id(f"dataset_{dataset_name}_{version.replace('.', '_')}")
    ds_id = _qualified("gen", ds_local)

    meta_local = run_id
    meta_id = _qualified("meta", meta_local)

    main_act_local = f"DatasetSplit_{run_id[:8]}"
    main_act_id = _qualified("blank", main_act_local)

    entities: dict[str, dict[str, Any]] = {}
    activities: dict[str, dict[str, Any]] = {}
    agents: dict[str, dict[str, Any]] = {agent_id: agent_props}
    used: dict[str, dict[str, Any]] = {}
    was_generated_by: dict[str, dict[str, Any]] = {}
    was_associated_with: dict[str, dict[str, Any]] = {}

    rel_counter = [0]

    def _blank_rel_id() -> str:
        rid = f"_:n{rel_counter[0]}"
        rel_counter[0] += 1
        return rid

    now = _iso_timestamp()

    # ── SOURCE DATASET ENTITY ──────────────────────────────
    entities[ds_id] = {
        "schema:name": [_typed_value_str(dataset_name)],
        "prov:type": [_qualified_name("sosa", "Sample")],
        "dct:description": [
            _typed_value_str(f"Source dataset {dataset_name} v{version}")
        ],
    }
    used[_blank_rel_id()] = {
        "prov:activity": run_act_id,
        "prov:entity": ds_id,
    }

    # ── SPLIT ENTITIES (output) ────────────────────────────
    split_entities: dict[str, str] = {}
    for split_name in split_stats:
        split_local = _safe_id(f"split_{split_name}_{run_id[:8]}")
        split_id = _qualified("gen", split_local)
        split_entities[split_name] = split_id

        stats = split_stats[split_name]
        entities[split_id] = {
            "schema:name": [_typed_value_str(f"{dataset_name} - {split_name}")],
            "prov:type": [_qualified_name("sosa", "Sample")],
            "dct:description": [
                _typed_value_str(
                    f"{split_name} split: {stats['num_samples']} samples, {stats['num_cases']} cases"
                )
            ],
        }

    # ── ACTIVITY ───────────────────────────────────────────
    run_activity: dict[str, Any] = {}
    run_activity["prov:type"] = [_qualified_name("schema", "Action")]
    run_activity["prov:startTime"] = [now]
    run_activity["prov:endTime"] = [now]
    run_activity["schema:name"] = [_typed_value_str(f"Split dataset {dataset_name}")]
    run_activity["gen:dataset_name"] = [_typed_value_str(dataset_name)]
    run_activity["gen:dataset_version"] = [_typed_value_str(version)]
    run_activity["gen:n_folds"] = [_typed_value_str(str(n_folds))]
    run_activity["gen:random_state"] = [_typed_value_str(str(random_state))]
    activities[run_act_id] = run_activity

    # ── CPM METADATA ENTITY ────────────────────────────────
    meta_entity: dict[str, list[Any]] = {}
    meta_entity["prov:type"] = [_qualified_name("cpm", "BundleMetadata")]
    meta_entity["gen:dataset_name"] = [_typed_value_str(dataset_name)]
    meta_entity["gen:dataset_version"] = [_typed_value_str(version)]
    meta_entity["gen:n_folds"] = [_typed_value_str(str(n_folds))]
    meta_entity["gen:random_state"] = [_typed_value_str(str(random_state))]
    meta_entity["gen:split_stats"] = [json.dumps(split_stats)]
    entities[meta_id] = meta_entity

    # ── CPM MAIN ACTIVITY ──────────────────────────────────
    main_activity: dict[str, Any] = {}
    main_activity["prov:type"] = [_qualified_name("cpm", "mainActivity")]
    main_activity["cpm:referencedMetaBundleId"] = [
        {"type": "prov:QUALIFIED_NAME", "$": meta_id},
    ]
    main_activity["dct:hasPart"] = [
        {"type": "prov:QUALIFIED_NAME", "$": run_act_id},
    ]
    activities[main_act_id] = main_activity

    # ── RELATIONSHIPS ──────────────────────────────────────
    was_associated_with[_blank_rel_id()] = {
        "prov:activity": run_act_id,
        "prov:agent": agent_id,
    }
    was_generated_by[_blank_rel_id()] = {
        "prov:entity": meta_id,
        "prov:activity": run_act_id,
    }

    # ── ASSEMBLE BUNDLE ────────────────────────────────────
    inner: dict[str, Any] = {"prefix": prefixes}
    if entities:
        inner["entity"] = entities
    if activities:
        inner["activity"] = activities
    if agents:
        inner["agent"] = agents
    if used:
        inner["used"] = used
    if was_generated_by:
        inner["wasGeneratedBy"] = was_generated_by
    if was_associated_with:
        inner["wasAssociatedWith"] = was_associated_with

    bundle_key = f"storage:{run_id}"
    return {"bundle": {bundle_key: inner}}


def _get_username() -> str:
    """Read username from active MLflow run tags."""
    try:
        client = mlflow.tracking.MlflowClient()  # pyright: ignore[reportPrivateImportUsage]
        active_run = mlflow.active_run()
        if active_run is None:
            return "unknown"
        run_id = active_run.info.run_id
        tags = {t.key: t.value for t in client.get_run(run_id).data.tags.values()}
    except Exception:
        tags = {}

    return tags.get("username", tags.get("mlflow.user", "unknown"))


def _typed_value_str(value: str) -> str:
    """Return a single-element list with the value (PROV-O convention)."""
    return str(value)
